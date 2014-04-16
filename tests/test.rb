#!/usr/bin/env ruby

require 'json'
require 'open3'

PRICER_BINARY = File.expand_path('../build/opencl_option_pricer')
DELTA = 1e-3

def run_test_case(input_parameters, expected_output)
  parameters_json = input_parameters.to_json

  result_json, errors, status = Open3.capture3(PRICER_BINARY, stdin_data: parameters_json)
  result = JSON.parse(result_json)

  expected_output['delta'] = 0 if expected_output['delta'].nil?

  error = nil

  if(expected_output['mean'])
    if(expected_output['precision'])
      precision = expected_output['precision']

      outcome = (result['mean'].round(precision) == expected_output['mean'].round(precision))
    else
      error = (result['mean'] - expected_output['mean'])
      outcome = error.abs < expected_output['delta']
    end
  else
    outcome = false
  end

  return [outcome, result['mean'], error]
end

@test_dir = File.expand_path(ARGV.shift)

stats = {
  tests: 0,
  passed: 0,
  failed: 0
}

@tests = Dir.glob(@test_dir + '/**/*.json').sort

@max_filename_length = @tests.map {|filename| filename.length - @test_dir.length - 1}.max

@tests.each do |test_filename|
  pretty_test_filename = test_filename.slice(@test_dir.length+1..-1)

  print "Running '#{pretty_test_filename}': "
  (@max_filename_length-pretty_test_filename.length).times {print ' '}
  stats[:tests] += 1

  input_parameters = JSON.parse(IO.read(test_filename))
  expected_output = input_parameters['expected']

  if(expected_output.nil?)
    puts 'MISSING SOLUTION'
    stats[:failed] += 1
    next
  end

  result,mean,error = run_test_case(input_parameters, expected_output)
  if result
    puts 'PASSED'
  else
    print "FAILED: #{mean}"
    print " / #{error}" unless error.nil?
    puts
  end

  stats[(result ? :passed : :failed)] += 1
end

puts
if(stats[:failed] == 0)
  puts "All tests passed!"
else
  puts "Failed test cases: #{stats[:failed]}/#{stats[:tests]}"
end
