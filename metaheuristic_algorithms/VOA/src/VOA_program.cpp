/*
	VOA_program.cpp: Created by Josue R. Cuevas

	Simple VOA code implemented on various benchmark functions
	with support to any dimensionality.
	For details please check out the README file
*/

/*
Copyright (C) <2017>  <Josue R. Cuevas>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "prototypes.h"

static uint32_t iterations=10000;

/*user helping routine*/
void usage(){
	printf("\n\nUsage: VOA_program.exe [-arg value ...]\n\n"
		"-h / -help / --h / --help: help menu printed as in here\n"
		"-stop_iterations: Iterations to be performed before stopping (Default=10000), it is an integer value\n"
		"-init_pop: initial population (default=10), it is an integer\n"
		"-n_strong: Number of viruses to be considered as strong (default=2), it is an integer\n"
		"-rate_strong: growing rate of strong viruses (default=2), it is an integer\n"
		"-rate_common: growing rate of common viruses (default=1), it is an integer\n"
		"-shift_value: Value used to shit the objective function (default=0.0), it is a float\n"
		"-function: name of the objective function value to be used (default=sphere), it is a string\n"
		"-debug_level: level of output information you want to see (default=debug), it is a string\n"
		"-dimensions: dimensionality of the problem (Default=10)"
		"\n\n\tExample: VOA_program.exe -init_pop 10 -n_strong 2 -rate_strong 5 -rate_common 2 -function ackley ...\n\n"
		"\nNOTE:  the options for the objective functions and debug levels are as follows:\n\n"
		"\tFunction Types: [sphere, rosenbrock, ackley, schwefel22, schwefel26, rastrigin, griewank, weierstrass]\n"
		"\tDebug Levels: [none, errors, warnings, debug, info, verbose]\n"
		"\n***** WARNING: debug levels after \"debug\" will cause a lot of output...!! use it with caution *****\n\n");
}


/*Parsing routine to be used in the program, to get user options*/
const char *functions[8]={"sphere", "rosenbrock", "ackley", "schwefel22",
	"schwefel26", "rastrigin", "griewank", "weierstrass"};
const char *debugging_level[6]={"none", "errors", "warnings", "debug", "info", "verbose"};

void parse_arguments(Virus_Population *current_population, int argc, char* argv[]){
	/*set parameters according to user inputs*/
	current_population->growing_rate_common = R_COMMON;
	current_population->growing_rate_strong = R_STRONG;
	current_population->intensity = 1;
	current_population->n_strong = N_STRONG_INIT;
	current_population->n_viruses = INIT_POP;
	current_population->n_viruses_to_remove = 0;
	current_population->remaining_after = INIT_POP;
	current_population->solutions = NULL;
	User_Options *options = current_population->usr_options;
	options->debugging_level = LOG_DEBUG;
	options->bench_function = options->function_to_use = SPHERE_F;
	options->dimensions = DIM;
	options->initial_population = INIT_POP;
	options->shifting_value = SHIFTING_VALUE;

	int i;

	/*parse all the options entered by user*/
	for(i=1;i<argc; i+=2){
		if(strcmp(argv[i], "-init_pop")==0){
			INIT_POP = atoi(argv[i+1]);
			current_population->n_viruses=INIT_POP;
			options->initial_population = INIT_POP;
		}else if(strcmp(argv[i], "-stop_iterations")==0){
			iterations = atoi(argv[i+1]);
		}else if(strcmp(argv[i], "-dimensions")==0){
			DIM = atoi(argv[i+1]);
			options->dimensions=DIM;
		}else if(strcmp(argv[i], "-n_strong")==0){
			N_STRONG_INIT = atoi(argv[i+1]);
			current_population->n_strong=N_STRONG_INIT;
		}else if(strcmp(argv[i], "-rate_strong")==0){
			R_STRONG = atoi(argv[i+1]);
			current_population->growing_rate_strong=R_STRONG;
		}else if(strcmp(argv[i], "-rate_common")==0){
			R_COMMON = atoi(argv[i+1]);
			current_population->growing_rate_common=R_COMMON;
		}else if(strcmp(argv[i], "-function")==0){
			if(strcmp(argv[i+1], "sphere")==0){
				options->function_to_use = options->bench_function = SPHERE_F;
			}else if(strcmp(argv[i+1], "rosenbrock")==0){
				options->function_to_use = options->bench_function = ROSENBROCK_F;
			}else if(strcmp(argv[i+1], "ackley")==0){
				options->function_to_use = options->bench_function = ACKLEY_F;
			}else if(strcmp(argv[i+1], "schwefel22")==0){
				options->function_to_use = options->bench_function = SCHWEFEL22_F;
			}else if(strcmp(argv[i+1], "schwefel26")==0){
				options->function_to_use = options->bench_function = SCHWEFEL26_F;
			}else if(strcmp(argv[i+1], "rastrigin")==0){
				options->function_to_use = options->bench_function = RASTRIGIN_F;
			}else if(strcmp(argv[i+1], "griewank")==0){
				options->function_to_use = options->bench_function = GRIEWANK_F;
			}else if(strcmp(argv[i+1], "weierstrass")==0){
				options->function_to_use = options->bench_function = WEIERSTRASS_F;
			}
		}else if(strcmp(argv[i], "-shift_value")==0){
			SHIFTING_VALUE = (float)atof(argv[i+1]);
			options->shifting_value = SHIFTING_VALUE;
		}else if(strcmp(argv[i], "-debug_level")==0){
			if(strcmp(argv[i+1], "none")==0){
				options->debugging_level = LOG_NONE;
			}else if(strcmp(argv[i+1], "errors")==0){
				options->debugging_level = LOG_ERRORS;
			}else if(strcmp(argv[i+1], "warnings")==0){
				options->debugging_level = LOG_WARNINGS;
			}else if(strcmp(argv[i+1], "debug")==0){
				options->debugging_level = LOG_DEBUG;
			}else if(strcmp(argv[i+1], "info")==0){
				options->debugging_level = LOG_INFO;
			}else if(strcmp(argv[i+1], "verbose")==0){
				options->debugging_level = LOG_VERBOSE;
			}else{
				LOGE("This debug level %s, is not supported", argv[i+1]);
			}
		}else{
			LOGE("This option is not recognized %s", argv[i]);
		}
	}

	if(options->debugging_level >= LOG_DEBUG){
		LOGD("Input options were:\n\n"
			"Initial population: %d\n"
			"Dimensions: %d\n"
			"Number of strong viruses: %d\n"
			"Growing rate of strong: %d\n"
			"Growing rate of common: %d\n"
			"Benchmark function: %s\n"
			"Shifting Value: %4.4f\n"
			"Debug Level: %s\n"
			"Iterations: %d\n\n", options->initial_population, options->dimensions,
			current_population->n_strong, current_population->growing_rate_strong,
			current_population->growing_rate_common, functions[options->bench_function],
			options->shifting_value, debugging_level[options->debugging_level-32], iterations);
	}
}

int _tmain(int argc, char* argv[]){
	/*generate containers for first population*/
	Virus_Population *current_population;
	User_Options *options;
	current_population = (Virus_Population*)malloc(sizeof(Virus_Population));
	options = (User_Options*)malloc(sizeof(User_Options));
	float Min_Obj_Fun=INF_VAL;
	float *temp_sol;

	current_population->usr_options = options;


	if(argc<2){
		/*no input we cannot continue*/
		LOGE("No input arguments, please try again ...");
		usage();
		return 0;
	}else if(strcmp((char*)argv[1], "-h")==0 || strcmp((char*)argv[1], "-help")==0 ||
		strcmp((char*)argv[1], "--h")==0 || strcmp((char*)argv[1], "--help")==0){
		/*check if user needs help*/
		LOGD("Printing Help option");
		usage();
		return 0;
	}else{
		/*parse input*/
		parse_arguments(current_population, argc, argv);
	}


	/*
		loop through the viruses until stopping criterion, this one
		can also be the achievement of the global minimum
	*/
	int it;
	float previous_avrg_obj=INF_VAL;
	float previous_min_val=INF_VAL;
	start_clock();

	/*Create first population*/
	if(create_first_population(current_population)!=SUCCESS){
		return FAILED_TO_CREATED_INITIAL_POPULATION;
	}


	/*estimate objective function value of all the viruses in pooled population*/
	temp_sol = (float*)malloc(sizeof(float)*options->dimensions);
	for(int l=0; l<current_population->n_viruses; l++){
		if(estimate_ObjFunc(current_population->solutions+l, current_population->usr_options)!=SUCCESS){
			if(options->debugging_level >= LOG_ERRORS){
				LOGE("Problem estimating objective function value for this population");
			}
			return CANT_CREATE_VIRUS;
		}

		if(l>current_population->n_strong){
			for(int k=0; k<current_population->n_strong; k++){
				if((current_population->solutions+l)->obj_func_val < (current_population->solutions+k)->obj_func_val){
					swap_solutions((current_population->solutions+k),
						(current_population->solutions+l), temp_sol, options->dimensions);
				}
			}
		}else{
			(current_population->solutions+l)->virus_type = STRONG_VIRUS;
		}
	}

	for(it=0;it<iterations;it++){
		/*create new viruses*/
		Virus_Population new_population;
		if(generate_new_solutions(current_population, &new_population)!=SUCCESS){
			if(options->debugging_level >= LOG_ERRORS){
				LOGE("Problem generating new viruses...");
			}
			return CANT_CREATE_VIRUS;
		}

		/*estimate objective function value of all the viruses in pooled population*/
		/*for(int l=0; l<new_population.n_viruses; l++){
			if(estimate_ObjFunc(new_population.solutions+l, current_population->usr_options)!=SUCCESS){
				if(options->debugging_level >= LOG_ERRORS){
					LOGE("Problem estimating objective function value for this population");
				}
				return CANT_CREATE_VIRUS;
			}
			if(Min_Obj_Fun>(new_population.solutions+l)->obj_func_val){
				Min_Obj_Fun = (new_population.solutions+l)->obj_func_val;
				memcpy(temp_sol, (new_population.solutions+l)->variable_val,
					sizeof(float)*options->dimensions);
			}
		}*/

		/*pool the population*/
		if(generate_pooled_population(current_population, &new_population)!=SUCCESS){
			if(options->debugging_level >= LOG_ERRORS){
				LOGE("Problem generating pooled population...");
			}
			return CANT_CREATE_VIRUS;
		}

		if(options->debugging_level >= LOG_DEBUG){
			LOGD("Rearranging viruses");
		}
		for(int l=0; l<current_population->n_viruses; l++){
			if(estimate_ObjFunc(current_population->solutions+l, current_population->usr_options)!=SUCCESS){
				if(options->debugging_level >= LOG_ERRORS){
					LOGE("Problem estimating objective function value for this population");
				}
				return CANT_CREATE_VIRUS;
			}
			if(Min_Obj_Fun>(current_population->solutions+l)->obj_func_val){
				Min_Obj_Fun = (current_population->solutions+l)->obj_func_val;
				memcpy(temp_sol, (current_population->solutions+l)->variable_val,
					sizeof(float)*options->dimensions);
			}

			if(options->debugging_level >= LOG_DEBUG){
				LOGD("%d, ", (current_population->solutions+l)->virus_type);
			}
		}



		/*estimate average objective function value*/
		if(estimate_avrg_obj(current_population)!=SUCCESS){
			if(options->debugging_level >= LOG_ERRORS){
				LOGE("Cannot estimate average objective function value...");
			}
			return CANT_CREATE_VIRUS;
		}

		if(options->debugging_level >= LOG_DEBUG){
			LOGD("Current population has %d members", current_population->n_viruses);
		}

		/*update intensity value if needed*/
		if(previous_avrg_obj==current_population->average_obj_func_value ||
			previous_min_val==Min_Obj_Fun){
			if(options->debugging_level >= LOG_DEBUG){
				LOGD("Updating the intensity value from %d to %d", current_population->intensity,
					current_population->intensity+1);
			}
			current_population->intensity++;
		}

		/*keep track of viruses min value*/
		previous_min_val=Min_Obj_Fun;

		if(previous_avrg_obj>current_population->average_obj_func_value){
			if(options->debugging_level >= LOG_DEBUG){
				LOGD("Average Obj Func Value is going to be updated: Old-> %4.4f, New-> %4.4f",
					previous_avrg_obj, current_population->average_obj_func_value);
			}
			previous_avrg_obj = current_population->average_obj_func_value;
		}

		/*determine viruses amount to delete*/
		current_population->n_viruses_to_remove =
			(uint32_t)(((float)rand()/RAND_MAX)*(current_population->n_viruses-current_population->n_strong));

		if(options->debugging_level >= LOG_DEBUG){
			LOGD("%d Members to be deleted ...", current_population->n_viruses_to_remove);
		}

		/*perform viruses contest*/
		if(population_contest(current_population)!=SUCCESS){
			if(options->debugging_level >= LOG_ERRORS){
				LOGE("Problem performing population contest...");
				return FAILED_TO_CREATED_NEW_POPULATION;
			}
		}

		/*perform population maintenance*/
		if(population_maintenance(current_population)!=SUCCESS){
			if(options->debugging_level >= LOG_ERRORS){
				LOGE("Problem performing population maintenance...");
				return FAILED_TO_CREATED_NEW_POPULATION;
			}
		}


		/*in case we passed the maximum number of viruses allowed*/
		if(current_population->n_viruses>=MAX_VIRUSES){
			if(options->debugging_level >= LOG_DEBUG){
				LOGD( "Second pass deletion, %d viruses in the pool", current_population->n_viruses);
			}


			/*determine viruses amount to delete*/
			current_population->n_viruses_to_remove =(current_population->n_viruses-options->initial_population);

			if(options->debugging_level >= LOG_DEBUG){
				LOGD("%d Members to be deleted ...", current_population->n_viruses_to_remove);
			}

			/*perform viruses contest*/
			if(population_contest(current_population)!=SUCCESS){
				if(options->debugging_level >= LOG_ERRORS){
					LOGE("Problem performing population contest...");
					return FAILED_TO_CREATED_NEW_POPULATION;
				}
			}

			/*perform population maintenance*/
			if(population_maintenance(current_population)!=SUCCESS){
				if(options->debugging_level >= LOG_ERRORS){
					LOGE("Problem performing population maintenance...");
					return FAILED_TO_CREATED_NEW_POPULATION;
				}
			}
		}
	}

	/*resume info*/
	printf("\n\n");
	LOGI("Time taken to perform optimization was: %4.15f seconds", end_query_clock());
	LOGI("Obj. Fun. Value: %4.4f", Min_Obj_Fun);
	printf("Solution found: [ ");
	for(it=0;it<options->dimensions;it++){
		printf("%4.4f, ", temp_sol[it]);
	}
	printf(" ]\n\n");

	/*remove used memory*/
	free_pop_viruses(current_population);
	free(current_population);
	free(options);
	free(temp_sol);

	system("pause");
	return SUCCESS;
}
