import argparse
import json
from string import Template

def str_to_file(filename, 
                input_str):
    try:
        with open(filename,'w') as f:
            f.write(input_str)
        f.close()
    except IOError:
        print('Unable to write '+filename)


def touch_submit(config_name):
    try:
        with open("touch.sh",'a') as f:
            f.write("touch " + config_name + "-{0..99}-sol.csv\n")
            f.write("touch " + config_name + "-{0..99}-log.csv\n")
        f.close()
    except IOError:
        print('Unable to write touch')
    try:
        with open("submit.sh",'a') as f:
            f.write("condor_submit " + config_name + ".sub\n")
        f.close()
    except IOError:
        print('Unable to write condor submit')



def get_job_spec(config_name):
    template = Template("""Universe       = vanilla
Requirements   = (Arch == "X86_64" ) && (OpSys == "LINUX") && (totalcpus >= 2)
RequestCPUs    = 4
Request_memory = 3G
Executable     = run.sh
Arguments      = "optimizer $$(Process) $alg_spec_file"
Output         = logs/output.$alg_spec_name.$$(Process).out
Error          = logs/error.$alg_spec_name.$$(Process).err
Log            = logs/log.$alg_spec_name.$$(Process).log
Should_transfer_files   = YES
Transfer_input_files    = $alg_spec_name-$$(Process)-sol.csv,$alg_spec_name-$$(Process)-log.csv,src/optimizer,src/$alg_spec_file,data/training_norm_outer_df.csv,data/testing_norm_outer_df.csv,data/cache.json
When_to_transfer_output = ON_EXIT
Transfer_output_files   = $alg_spec_name-$$(Process)-sol.csv,$alg_spec_name-$$(Process)-log.csv,cache.json
Transfer_output_remaps  = "cache.json=$alg_spec_name.cache.$$(Process).json"
Queue 100 
""")
    file_str = template.substitute(alg_spec_file=config_name + ".json", 
                       alg_spec_name=config_name)
    str_to_file(config_name + ".sub", file_str)


def get_alg_spec(cache_key, 
                 suffix="u", 
                 kernel_func_str = "random_uniform",
                 recurrent_func_str = "random_uniform",
                 bias_func_str = "random_uniform"):
    template = Template("""{
"config_name" : "$name",
"data_folder" : "./",
"results_folder":"./",
"cache_file" : "./cache.json",
"optimizer_class" : "algorithms.RandomSearchSpecificArch",
"data_reader_class" : "util.ParkingDFDataReader",
"blind" : 1,
"x_features" : ["Others-CCCPS8", "Others-CCCPS98", "Shopping", "BHMBCCMKT01",
       "BHMBCCPST01", "BHMBCCSNH01", "BHMBCCTHL01", "BHMBRCBRG01",
       "BHMBRCBRG02", "BHMBRCBRG03", "BHMEURBRD01", "BHMEURBRD02",
       "BHMMBMMBX01", "BHMNCPHST01", "BHMNCPLDH01", "BHMNCPNHS01",
       "BHMNCPNST01", "BHMNCPPLS01", "BHMNCPRAN01", "Broad Street",
       "Bull Ring", "NIA Car Parks", "NIA South",
       "Others-CCCPS105a", "Others-CCCPS119a", "Others-CCCPS133",
       "Others-CCCPS135a", "Others-CCCPS202", "weekday", "time"],
"y_features" : ["Others-CCCPS8", "Others-CCCPS98", "Shopping", "BHMBCCMKT01",
       "BHMBCCPST01", "BHMBCCSNH01", "BHMBCCTHL01", "BHMBRCBRG01",
       "BHMBRCBRG02", "BHMBRCBRG03", "BHMEURBRD01", "BHMEURBRD02",
       "BHMMBMMBX01", "BHMNCPHST01", "BHMNCPLDH01", "BHMNCPNHS01",
       "BHMNCPNST01", "BHMNCPPLS01", "BHMNCPRAN01", "Broad Street",
       "Bull Ring", "NIA Car Parks", "NIA South",
       "Others-CCCPS105a", "Others-CCCPS119a", "Others-CCCPS133",
       "Others-CCCPS135a", "Others-CCCPS202"],
"architecture": [$architecture],
"max_look_back" : $look_back,
"min_look_back" : $look_back,
"params_neuron" : 4,
"max_evals": 100,
"targets": [-1.0],
"metrics": ["mae"],
"kernel_init_func": "algorithms.$kernel_func",
"recurrent_init_func": "algorithms.$recurrent_func",
"bias_init_func": "algorithms.$bias_func"
}""")
    config_name = cache_key + "." + suffix
    architecture_str = cache_key[3:cache_key.find("-28.")].replace("-",",")
    look_back_str = cache_key[cache_key.find("-28.")+4:]
    look_back_str = look_back_str[:look_back_str.find(".")]    
    file_str = template.substitute(name=config_name, 
                        architecture=architecture_str,
                        look_back=look_back_str, 
                        kernel_func=kernel_func_str, 
                        recurrent_func=recurrent_func_str, 
                        bias_func=bias_func_str)
    str_to_file(config_name + ".json", file_str)
    # Generate the job description for Condor HT
    get_job_spec(config_name)
    # Append to touch and submit scripts, to ease execution
    touch_submit(config_name)


def load_from_file(filename):        
    cache = {}
    try:
        with open(filename, 'r') as f:
            f_str = f.read()
            cache = json.loads(f_str)
            print(str(len(cache)) + ' entries loaded into the cache memory')
        f.close()
    except IOError:
        print('Unable to load the cache')
    return cache







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--cache',
          type=str,
          default="cache.json",
          help='Cache file name.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    cache = load_from_file( FLAGS.cache )
    for cache_key in cache.keys():
        get_alg_spec(cache_key)

