#!/usr/bin/env ruby

require 'json'
require 'open3'
require 'csv'

PRICER_BINARY = File.expand_path(__FILE__ + '/../../build/opencl_option_pricer')
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

def test_case_to_csv(input_parameters)
  strike_price = input_parameters['strike_price']

  volatility = input_parameters['volatility']
  if(volatility.is_a? Array)
    volatility = '\specialcell{' + volatility.join(' \\\\ ') + '}'
  end

  correlation = input_parameters['correlation'] ? input_parameters['correlation'][0][1] : nil
  averaging_steps = input_parameters['averaging_steps']

  type = case input_parameters['type']
         when 'european' then 'European'
         when 'asian_geometric' then 'Geometric Asian'
         when 'asian_arithmetic' then 'Arithmetic Asian'
         when 'basket_geometric' then 'Geometric Basket'
         when 'basket_arithmetic' then 'Arithmetic Basket'
         end

  direction = input_parameters['direction'] == 'call' ? 'Call' : 'Put'

  control_variate = case input_parameters['control_variate']
                    when 'none' then 'N'
                    when 'geometric' then 'G'
                    when 'geometric_adjusted_strike' then 'GA'
                    end

  return [type, direction, control_variate, strike_price, volatility, correlation, averaging_steps]
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

  if(@csv)
    line = test_case_to_csv(input_parameters)
    mean_string = '%g' % ('%.05f' % mean)
    line << mean_string

    @csv << line
  end
end

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
