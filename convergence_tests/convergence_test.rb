#!/usr/bin/env ruby

require 'json'
require 'open3'
require 'csv'
require 'pp'

PRICER_BINARY = File.expand_path('../build/opencl_option_pricer')
DELTA = 1e-3
SAMPLES = {:none => [10000, 100000, 1000000], :geometric => [10000, 100000, 1000000]}

def run_pricer(input_parameters)
  parameters_json = input_parameters.to_json

  result_json, errors, status = Open3.capture3(PRICER_BINARY, stdin_data: parameters_json)
  result = JSON.parse(result_json)

  return result
end

def run_convergence_test(input_parameters, tests)
  results = {}

  tests.each do |control_variate, samples|
    results[control_variate] = {}

    input_parameters['control_variate'] = control_variate.to_s

    samples.each do |sample_count|
      input_parameters['samples'] = sample_count

      result = run_pricer(input_parameters)

      results[control_variate][sample_count] = result
      print '.'
    end
  end

  best_result = results[:geometric].sort_by {|sample_count, result| sample_count}.last[1]['mean']
  $stderr.puts "BEST RESULT: #{best_result}"
  PP.pp(results, $stderr)

  results.each do |control_variate, cv_results|
    last_confidence_interval_size = nil
    last_mean_difference = nil

    cv_results.sort_by {|sample_count, result| sample_count}.each do |sample_count, result|
      confidence_interval_size = result['confidence_interval'][1] - result['confidence_interval'][0]
      mean_difference = (result['mean'] - best_result).abs
      
      if(last_confidence_interval_size and not confidence_interval_size < last_confidence_interval_size)
        puts "FAILED: confidence interval size not converging"
        return false
      end
      if(last_mean_difference and not mean_difference < last_mean_difference)
        puts "FAILED: means not converging"
        return false
      end
      
      last_confidence_interval_size = confidence_interval_size
      last_mean_difference = mean_difference
    end
  end

  puts "PASSED"
  return true
end

@test_dir = File.expand_path(ARGV.shift)
@csv_file = ARGV.shift unless ARGV.empty?

stats = {
  tests: 0,
  passed: 0,
  failed: 0
}

@tests = Dir.glob(@test_dir + '/**/*.json').sort

@max_filename_length = @tests.map {|filename| filename.length - @test_dir.length - 1}.max

if(@csv_file)
  @csv = CSV.open(@csv_file, 'wb')
  @csv << ['Type', 'Direction', 'Control Variate', 'Strike Price', 'Volatility', 'Correlation', 'Averaging Steps', 'Price']
end

@tests.each do |test_filename|
  pretty_test_filename = test_filename.slice(@test_dir.length+1..-1)

  print "Running '#{pretty_test_filename}': "
  (@max_filename_length-pretty_test_filename.length).times {print ' '}
  stats[:tests] += 1

  begin
    input_parameters = JSON.parse(IO.read(test_filename))
  rescue JSON::ParserError => e
    puts "PARSING FAILED: #{e}"
    stats[:failed] += 1
    next
  end

  result = run_convergence_test(input_parameters, SAMPLES)

  stats[(result ? :passed : :failed)] += 1

  # if(@csv)
  #   line = test_case_to_csv(input_parameters)
  #   mean_string = '%g' % ('%.05f' % mean)
  #   line << mean_string

  #   @csv << line
  # end
end
exit 0

puts
if(stats[:failed] == 0)
  puts "All tests passed!"
else
  puts "Failed test cases: #{stats[:failed]}/#{stats[:tests]}"
end

if(@csv)
  @csv.close
  puts "CSV written to #{@csv_file}"
end
