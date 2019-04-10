# frozen_string_literal: true

require 'parallel'
require 'thor'
require 'tmpdir'

def concat_audio(input_paths, output_path)
  File.open(output_path, 'wb') do |output|
    input_paths.each_slice(32).each do |slice|
      input_data = Parallel.map(slice, in_threads: 16) do |input_path|
        Dir.mktmpdir do |dir|
          warn input_path

          case input_path
          when /\.(txt|jpg|jpeg|png|json)/
            next nil
          when /\.raw$/
            pcm_path = input_path
          else
            pcm_path = "#{dir}/output.raw"
            ffmpeg_output = `ffmpeg -i #{input_path} -ac 1 -ar 48000 -f s16le -acodec pcm_s16le #{pcm_path} 2>&1`
            unless File.exist?(pcm_path) && File.size(pcm_path).positive?
              warn "failed #{input_path} #{ffmpeg_output}"
              next nil
            end
          end

          normalized_path = "#{dir}/normalized.raw"
          sox_output = `sox -r 48000 -e signed -c 1 -b 16 #{pcm_path} #{normalized_path} gain -n 2>&1`

          if File.exist?(normalized_path) && File.size(normalized_path).positive?
            File.read(normalized_path)
          else
            warn "failed #{input_path} #{sox_output}"
            next nil
          end
        end
      end
      input_data.each do |data|
        output.write(data) if data
      end
    end
  end
end

def denoise_training_path
  if File.exists?('bin/Release/denoise_training')
    'bin/Release/denoise_training'
  else
    'bin/denoise_training'
  end
end

class MyCLI < Thor
  desc 'prepare_pcm', 'prepare raw pcm from directory'
  option :input, required: true, desc: 'dir path containing clean audio'
  option :output, required: true, desc: 'output raw pcm path'
  def prepare_pcm
    concat_audio(Dir.glob("#{options[:input]}/**/*.*").sort, options[:output])
  end

  desc 'prepare_vec', 'prepare '
  option :output_count, required: true, desc: 'output frame count'
  def prepare_vec
    `#{denoise_training_path} --clean clean.raw --noise noise.raw --output /tmp/unused --output_count #{options[:output_count]} > output.f32`
    `pipenv run python training/bin2hdf5.py output.f32 -1 87 denoise_data9.h5`
  end

  desc 'manual', 'show manual'
  def manual
    warn <<EOS
# get noise data
curl -L https://people.xiph.org/~jm/demo/rnnoise/rnnoise_contributions.tar.gz | tar zx -C /tmp

# get speech data
curl -L http://www-mmsp.ece.mcgill.ca/Documents/Data/TSP-Speech-Database/48k.zip > /tmp/48k.zip
unzip -d /tmp /tmp/48k.zip

# setup
bundle install
pipenv install

# prepare training data
bundle exec ruby experiment1.rb prepare_pcm --input /tmp/48k --output clean.raw
bundle exec ruby experiment1.rb prepare_pcm --input /tmp/rnnoise_contributions --output noise.raw
bundle exec ruby experiment1.rb prepare_vec --output-count 50000

# train
pipenv run python training/rnn_train.py

# use gpu in docker container

EOS
  end
end

MyCLI.start(ARGV)
