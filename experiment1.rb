# frozen_string_literal: true

require 'parallel'
require 'thor'
require 'tmpdir'

def concat_audio(input_paths, output_path)
  File.open(output_path, 'wb') do |output|
    input_paths.each_slice(16).each do |slice|
      input_data = Parallel.map(slice, in_threads: 8) do |input_path|
        warn input_path

        case input_path
        when /\.(txt|jpg|jpeg|png|json)/
          nil
        when /\.raw$/
          File.read(input_path)
        else
          Dir.mktmpdir do |dir|
            tmp_path = "#{dir}/output.raw"
            ffmpeg_output = `ffmpeg -i #{input_path} -ac 1 -ar 48000 -f s16le -acodec pcm_s16le #{tmp_path} 2>&1`
            if File.exist?(tmp_path) && File.size(tmp_path) > 0
              File.read(tmp_path)
            else
              warn "failed #{input_path} #{ffmpeg_output}"
              nil
            end
          end
        end
      end
      input_data.each do |data|
        output.write(data) if data
      end
    end
  end
end

class MyCLI < Thor
  desc 'prepare_pcm', 'prepare raw pcm from directory'
  option :input, required: true, desc: 'dir path containing clean audio'
  option :output, required: true, desc: 'output raw pcm path'
  def prepare_pcm
    concat_audio(Dir.glob("#{options[:input]}/*.*").sort, options[:output])
  end

  desc 'prepare_vec', 'prepare '
  option :output_count, required: true, desc: 'output frame count'
  def prepare_vec
    `pipenv run python src/denoise_training --clean clean.raw --noise noise.raw --output /tmp/unused --output_count #{options { :output_count }} > output.f32`
    `pipenv run python training/bin2hdf5.py output.f32 -1 87 denoise_data9.h5`
  end

  desc 'train', ''
  def train
    warn 'execute following'
    puts 'pipenv run ~/rnnoise/training/rnn_train.py'
  end
end

MyCLI.start(ARGV)
