# frozen_string_literal: true

require 'json'
require 'open3'
require 'parallel'
require 'shellwords'
require 'thor'
require 'tmpdir'

def ffprobe(path)
  JSON.parse(`ffprobe -v quiet -print_format json -show_format -show_streams #{path}`)
end

def find_audio_stream(probe_result)
  probe_result['streams'].find { |stream| stream['codec_type'] == 'audio' }
end

def exec_command(command)
  stdout, status = Open3.capture2(command)
  raise "failed_command:#{command}\tstatus:#{status}" unless status.success?

  stdout
end

def concat_audio(input_paths, output_path, max_sec:)
  output_samples = 0
  max_samples = (48_000 * max_sec).to_i

  File.open(output_path, 'wb') do |output|
    input_paths.each_slice(32).each do |slice|
      process_file = lambda do |input_path|
        Dir.mktmpdir do |dir|
          warn input_path
          raw_path = "#{dir}/output.raw"

          case input_path
          when /\.(txt|jpg|jpeg|png|json|md)/
            return []
          when /\.zip$/
            exec_command("unzip -d #{dir}/extracted #{input_path} 2>&1")
            return Dir.glob("#{dir}/extracted/**/*.*").sort.map { |path| process_file.call(path) }
          when /\.raw$/
            # normalize
            pcm_path = input_path
            exec_command("sox -r 48000 -e signed -c 1 -b 16 #{pcm_path} #{raw_path} gain -n 2>&1")
          else
            # convert to 32bit float wav
            float_path = "#{dir}/float.wav"
            exec_command("ffmpeg -i #{input_path} -ar 48000 -f wav -acodec pcm_f32le #{float_path} 2>&1")

            # split channel + normalize + remove silence
            probe_result = ffprobe(input_path)
            channels = find_audio_stream(probe_result)['channels']
            (1..channels).each do |channel_idx|
              exec_command("sox #{float_path} #{dir}/soxtmp1.wav remix #{channel_idx} 2>&1")
              exec_command("sox #{dir}/soxtmp1.wav #{dir}/soxtmp2.wav silence 1 1.0 0.1% 1 1.0 0.1% : restart 2>&1")
              exec_command("sox #{dir}/soxtmp2.wav #{dir}/splitted#{channel_idx}.wav gain -n 2>&1")
            end

            # concat + convert to raw
            inputs = (1..channels).map do |channel_idx|
              "#{dir}/splitted#{channel_idx}.wav"
            end
            exec_command("sox #{inputs.join(' ')} -e signed -b 16 #{raw_path} 2>&1")
          end

          File.read(raw_path)
        end
      rescue StandardError => e
        warn "failed #{input_path} because #{e}"
      end

      input_data = Parallel.map(slice, in_threads: 16) do |input_path|
        process_file.call(input_path)
      end
      input_data.flatten.each do |data|
        if max_sec >= 0 && data.length > 2 * (max_samples - output_samples)
          data = data[0, 2 * (max_samples - output_samples)]
        end
        output.write(data)
        output_samples += data.length / 2
        return if max_sec >= 0 && output_samples >= max_samples
      end
    end
  end
end

def denoise_training_path
  if File.exist?('bin/Release/denoise_training')
    'bin/Release/denoise_training'
  else
    'bin/denoise_training'
  end
end

class MyCLI < Thor
  desc 'prepare_pcm', 'prepare raw pcm from directory'
  option :input, required: true, desc: 'dir path containing clean audio'
  option :output, required: true, desc: 'output raw pcm path'
  option :max_sec, type: :numeric, required: false, default: -1, desc: 'output raw pcm max length'
  def prepare_pcm
    concat_audio(Dir.glob("#{options[:input]}/**/*.*").sort, options[:output], max_sec: options[:max_sec])
  end

  desc 'interleave_pcm', 'blend multiple raw pcm'
  option :input, required: true, desc: 'comma separated raw audio path'
  option :output, required: true, desc: 'output raw pcm path'
  option :interleave_sec, type: :numeric, required: true, desc: 'interleave interval'
  option :max_sec, type: :numeric, required: false, default: -1, desc: 'output raw pcm max length'
  def interleave_pcm
    block_size = 2 * (48_000 * options[:interleave_sec]).to_i
    inputs = options[:input].split(',')
    offset = 0
    output_size = 0
    max_output_size = 2 * (48_000 * options[:max_sec]).to_i
    File.open(options[:output], 'wb') do |output|
      until inputs.empty?
        (0..inputs.length - 1).each do |i|
          data = File.binread(inputs[i], block_size, offset)
          if data && !data.empty?
            if max_output_size >= 0 && output_size + data.length > max_output_size
              data = data[0, max_output_size - output_size]
            end
            output.write(data)
            output_size += data.length
            return if max_output_size >= 0 && output_size > max_output_size
          else
            inputs[i] = nil
          end
        end
        offset += block_size
        inputs = inputs.compact
      end
    end
  end

  option :interleave_sec, required: false, default: -1, desc: 'input interleave interval'

  desc 'prepare_vec', 'prepare '
  option :output_count, required: true, desc: 'output frame count'
  def prepare_vec
    `#{denoise_training_path} --clean clean.raw --noise noise.raw --output /tmp/unused --output_count #{options[:output_count]} > output.f32`
    `pipenv run python training/bin2hdf5.py output.f32 -1 87 denoise_data9.h5`
  end

  desc 'manual', 'show manual'
  def manual
    warn <<~EOS
      # get noise data
      curl -L https://people.xiph.org/~jm/demo/rnnoise/rnnoise_contributions.tar.gz | tar zx -C /mldata1

      # get speech data
      curl -L http://www-mmsp.ece.mcgill.ca/Documents/Data/TSP-Speech-Database/48k.zip > /mldata1/48k.zip
      unzip -d /mldata1 /mldata1/48k.zip

      # setup
      bundle install
      pipenv install

      # prepare training data
      bundle exec ruby experiment1.rb prepare_pcm --input /mldata1/48k --output clean.raw
      bundle exec ruby experiment1.rb prepare_pcm --input /mldata1/rnnoise_contributions --output noise.raw
      bundle exec ruby experiment1.rb prepare_vec --output-count 50000

      # train
      pipenv run python training/rnn_train.py

      # use gpu in docker container

    EOS
  end
end

MyCLI.start(ARGV)
