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

/*call this one always when setting parameters*/
void reset_functions(User_Options *options){
	SCHWEFEL26_GM = options->dimensions*(-418.9829f);
}

/*
	Routine to create first population of viruses
	in a random mode, using the [min, max] values
*/
static void generate_solution(Virus *candidate, uint32_t dimensions, float MaxMin[2], uint8_t debug_l){
	int i;
	float maxVal, minVal, range;
	maxVal = MaxMin[1];
	minVal = MaxMin[0];
	range = maxVal-minVal;

	if(debug_l>=LOG_VERBOSE){
		printf("[ ");
	}
	for(i=0;i<dimensions;i++){
		candidate->variable_val[i] = minVal + ((float)rand()/RAND_MAX)*range;
		if(debug_l>=LOG_VERBOSE){
			printf("%4.4f, ", candidate->variable_val[i]);
		}
	}
	if(debug_l>=LOG_VERBOSE){
		printf(" ] -> ");
	}
}

static uint8_t get_range(uint8_t function, float MaxMin[2]){
	switch(function){
		case SPHERE_F:
		{
			MaxMin[0] = SPHERE_RANGE[0];
			MaxMin[1] = SPHERE_RANGE[1];
			break;
		}
		case ROSENBROCK_F:
		{
			MaxMin[0] = ROSENBROCK_RANGE[0];
			MaxMin[1] = ROSENBROCK_RANGE[1];
			break;
		}
		case ACKLEY_F:
		{
			MaxMin[0] = ACKLEY_RANGE[0];
			MaxMin[1] = ACKLEY_RANGE[1];
			break;
		}
		case SCHWEFEL22_F:
		{
			MaxMin[0] = SCHWEFEL22_RANGE[0];
			MaxMin[1] = SCHWEFEL22_RANGE[1];
			break;
		}
		case SCHWEFEL26_F:
		{
			MaxMin[0] = SCHWEFEL26_RANGE[0];
			MaxMin[1] = SCHWEFEL26_RANGE[1];
			break;
		}
		case RASTRIGIN_F:
		{
			MaxMin[0] = RASTRIGIN_RANGE[0];
			MaxMin[1] = RASTRIGIN_RANGE[1];
			break;
		}
		case GRIEWANK_F:
		{
			MaxMin[0] = GRIEWANK_RANGE[0];
			MaxMin[1] = GRIEWANK_RANGE[1];
			break;
		}
		case WEIERSTRASS_F:
		{
			MaxMin[0] = WEIERSTRASS_RANGE[0];
			MaxMin[1] = WEIERSTRASS_RANGE[1];
			break;
		}
		default:
		{
			LOGW("This function is not supported: %d, exiting ...", function);
			return WRONG_BENCHMARK_FUNCTION;
		}
	}

	return SUCCESS;
}

uint8_t create_first_population(Virus_Population *viruses){
	User_Options *options = viruses->usr_options;
	Virus *candidates = NULL;
	uint32_t n_viruses = options->initial_population;
	uint8_t function_to_use = options->bench_function;
	uint32_t dimensions = options->dimensions;
	float MaxMin[2];

	int i;

	if(options->debugging_level>=LOG_DEBUG){
		LOGD("Creating first population of viruses...");
	}

	candidates = (Virus*)malloc(sizeof(Virus)*n_viruses);
	if(!candidates){
		if(options->debugging_level>=LOG_ERRORS){
			LOGE("Cannot create initial population, not enough memory..!!");
		}
		return FAILED_TO_CREATED_INITIAL_POPULATION;
	}

	if(get_range(function_to_use, MaxMin)!=SUCCESS){
		if(options->debugging_level>=LOG_ERRORS){
			LOGE("Cannot get the MinMax range for the OBJ FUNCTION..!!");
		}
		return FAILED_TO_CREATED_INITIAL_POPULATION;
	}

	if(options->debugging_level>=LOG_DEBUG){
		LOGD("The function use has a range of [%4.4f, %4.4f]", MaxMin[0], MaxMin[1]);
	}


	for(i=0;i<n_viruses;i++){
		candidates[i].variable_val = NULL;
		candidates[i].variable_val = (float*)malloc(sizeof(float)*dimensions);
		if(!candidates[i].variable_val){
			if(options->debugging_level>=LOG_ERRORS){
				LOGE("Cannot create initial population, not enough memory..!!");
			}
			return FAILED_TO_CREATED_INITIAL_POPULATION;
		}
		generate_solution(candidates+i, dimensions, MaxMin, options->debugging_level);
		candidates[i].obj_func_val=9999999999.99f;
		candidates[i].remove_virus=FALSE;
		candidates[i].virus_type=COMMON_VIRUS;//all are marked at common

		if(options->debugging_level>=LOG_VERBOSE){
			printf("<type: %d, removal_flag: %d>\n", candidates[i].virus_type,
				candidates[i].remove_virus);
		}
	}

	viruses->solutions = candidates;

	return SUCCESS;
}


/*
	Reproduce the virus according to its growing rate, this
	is done for a single virus
*/
void mutate_virus(Virus *old_candidate, Virus *new_candidate, User_Options *options,
				  float MinMax[2], uint32_t dimensions, uint32_t intensity){
	int i;
	float maxVal, minVal, range;
	float perturbation;
	maxVal = MinMax[1];
	minVal = MinMax[0];
	range = maxVal-minVal;

	if(options->debugging_level>=LOG_VERBOSE){
		printf("[ ");
	}
	for(i=0;i<dimensions;i++){
		perturbation = ((((float)rand()/RAND_MAX)-((float)rand()/RAND_MAX))*(range)/intensity);
		new_candidate->variable_val[i] = old_candidate->variable_val[i] + perturbation;
		if(new_candidate->variable_val[i]>maxVal || new_candidate->variable_val[i]<minVal){
			//regenerate since is out of bound
			new_candidate->variable_val[i] = minVal + (rand()/RAND_MAX)*range;
		}
		if(options->debugging_level>=LOG_VERBOSE){
			printf("%4.4f, ", new_candidate->variable_val[i]);
		}
	}
	if(options->debugging_level>=LOG_VERBOSE){
		printf(" ]");
	}
}

uint8_t generate_new_solutions(Virus_Population *current_viruses, Virus_Population *new_viruses){
	User_Options *options = current_viruses->usr_options;
	Virus *candidates = current_viruses->solutions;
	Virus *new_candidates = NULL;
	uint8_t rate_strong = current_viruses->growing_rate_strong;
	uint8_t rate_common = current_viruses->growing_rate_common;
	uint32_t current_pop_size = current_viruses->n_viruses;
	uint32_t n_strong = current_viruses->n_strong;
	uint32_t n_common = current_pop_size-n_strong;
	uint32_t new_pop_size = n_strong*rate_strong + n_common*rate_common;
	uint32_t dimensions = options->dimensions;
	uint32_t intensity = current_viruses->intensity;
	float MinMax[2];
	uint32_t i;

	/*to regenerate values if the are out of range*/
	get_range(options->bench_function, MinMax);

	/*using information*/
	new_viruses->n_viruses = new_pop_size;

	/*creating memory needed*/
	new_candidates = (Virus*)malloc(sizeof(Virus)*new_pop_size);
	if(!new_candidates){
		LOGE("Cannot generate new solutions, not enough memory !!");
		return FAILED_TO_CREATED_NEW_POPULATION;
	}

	uint32_t j=0;
	for(i=0; i<new_pop_size; i++){
		new_candidates[i].remove_virus=FALSE;
		new_candidates[i].virus_type=COMMON_VIRUS;
		new_candidates[i].obj_func_val = 999999999.99f;
		new_candidates[i].variable_val = NULL;
		new_candidates[i].variable_val = (float*)malloc(sizeof(float)*dimensions);
		if(!new_candidates[i].variable_val){
			LOGE("Cannot generate new solutions, not enough memory !!");
			return FAILED_TO_CREATED_NEW_POPULATION;
		}


		mutate_virus(candidates+j, new_candidates+i, options,
			MinMax, dimensions, intensity);
		if(options->debugging_level>=LOG_VERBOSE){
			printf(" --> type: %d\n", (new_candidates+i)->virus_type);
		}
		if(i>0 && j<n_strong){
			if((i%rate_strong)==0){
				j++;
			}
		}else if(i>0){
			if((i%rate_common)==0){
				j++;
			}
			intensity=1;//since we dont need this for common viruses
		}
	}

	new_viruses->solutions = new_candidates;

	return SUCCESS;
}

/*
	Perform population pooling, to combine the predecessors and
	newly generated members
*/
uint8_t generate_pooled_population(Virus_Population *pooled_viruses,
								   Virus_Population *new_viruses){
	User_Options *options = pooled_viruses->usr_options;
	Virus *candidates = pooled_viruses->solutions;
	Virus *new_candidates = new_viruses->solutions;
	Virus *pooled_candidates=NULL;
	uint32_t pooled_pop_size = pooled_viruses->n_viruses
		+ new_viruses->n_viruses;
	uint32_t i=0, j=0, k=0;
	uint32_t dimensions = options->dimensions;

	/*pooled population*/
	pooled_candidates = (Virus*)malloc(sizeof(Virus)*pooled_pop_size);
	for(i=0;i<pooled_pop_size;i++){
		if(i<pooled_viruses->n_viruses){
			if(create_virus(pooled_candidates+i, candidates+j, dimensions)!=SUCCESS){
				return CANT_CREATE_VIRUS;
			}
			j++;
		}else{
			if(create_virus(pooled_candidates+i, new_candidates+k, dimensions)!=SUCCESS){
				return CANT_CREATE_VIRUS;
			}
			k++;
		}
	}

	/*remove unnecessary memory*/
	free(new_viruses->solutions);
	free(pooled_viruses->solutions);

	/*link the viruses*/
	pooled_viruses->solutions = pooled_candidates;
	pooled_viruses->n_viruses = pooled_pop_size;

	return SUCCESS;
}



/*
	Estimates the objective function value according to
	dimensionality and shifting values to speed up things
	we use registers and we process one virus at a time
	so we can put it on RAM memory right away, locally
	in this function.
*/
uint8_t estimate_ObjFunc(Virus *candidate, User_Options *options){
	int i, j;
	float x_0, x_1, const_val1, const_val2;
	float temp1, temp2;
	const float PI_VAL = 3.14159265358979323846f;
	const float a=0.5f, b=3.0f;
	int kmax=20;

	/*Determine which function is to be used*/
	switch(options->function_to_use){
		case SPHERE_F:
		{
			candidate->obj_func_val=0.0f;
			for(i=0; i<options->dimensions; i++){
				x_0 = candidate->variable_val[i];
				candidate->obj_func_val += x_0*x_0;
			}
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("SPHERE_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		case ROSENBROCK_F:
		{
			candidate->obj_func_val=0.0f;
			for(i=0; i<options->dimensions-1; i++){
				x_0 = candidate->variable_val[i];
				x_1 = candidate->variable_val[i+1];
				candidate->obj_func_val += 100*(
					(x_1-(x_0*x_0))*(x_1-(x_0*x_0))
					+ (x_0-1.0f)*(x_0-1.0f) );
			}
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("ROSENBROCK_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		case ACKLEY_F:
		{
			candidate->obj_func_val=0.0f;
			temp1=0.0f;
			temp2=0.0f;
			const_val1 = 1.0f / (float)options->dimensions;
			const_val2 = expf(1.0f);
			for(i=0; i<options->dimensions; i++){
				x_0 = candidate->variable_val[i];
				temp1 += x_0*x_0;
				temp2 += cosf(2.0f*PI_VAL*x_0);
			}
			temp1 *= const_val1;
			temp1 = powf(temp1, 0.2f);
			temp1 = -20.0f*expf(-temp1)+20.0f;
			temp2 *= const_val1;
			temp2 = -expf(temp2)+const_val2;
			candidate->obj_func_val = temp1 + temp2;
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("ACKLEY_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		case SCHWEFEL22_F:
		{
			candidate->obj_func_val=0.0f;
			temp1=0.0f;
			temp2=1.0f;
			for(i=0; i<options->dimensions; i++){
				x_0 = candidate->variable_val[i];
				temp1 += fabs(x_0);
				temp2 *= fabs(x_0);
			}
			candidate->obj_func_val = temp1 + temp2;
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("SCHWEFEL22_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		case SCHWEFEL26_F:
		{
			candidate->obj_func_val=0.0f;
			for(i=0; i<options->dimensions; i++){
				x_0 = candidate->variable_val[i];
				candidate->obj_func_val += x_0*sinf(sqrt(fabs(x_0)));
			}
			candidate->obj_func_val = -candidate->obj_func_val;
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("SCHWEFEL26_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		case RASTRIGIN_F:
		{
			candidate->obj_func_val=0.0f;
			for(i=0; i<options->dimensions; i++){
				x_0 = candidate->variable_val[i];
				candidate->obj_func_val += x_0*x_0 - 10.0f*cosf(2.0f*PI_VAL*x_0) + 10.0f;
			}
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("RASTRIGIN_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		case GRIEWANK_F:
		{
			candidate->obj_func_val=0.0f;
			temp1=0.0f; temp2=1.0f;
			for(i=0; i<options->dimensions; i++){
				x_0 = candidate->variable_val[i];
				temp1 += x_0*x_0;
				temp2 *= cosf(x_0/sqrt((float)i));
			}
			temp1 *= 1.0f/4000.0f;
			candidate->obj_func_val = temp1 - temp2 + 1.0f;
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("GRIEWANK_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		case WEIERSTRASS_F:
		{
			candidate->obj_func_val=0.0f;
			temp1=0.0f; temp2=0.0f;
			for(i=0; i<options->dimensions; i++){
				x_0 = candidate->variable_val[i];
				for(j=0;j<kmax; j++){
					temp1 += pow(a, j)*cosf(2.0f*PI_VAL*pow(b, j)*(x_0+0.5f));
				}
			}
			for(j=0;j<kmax; j++){
				temp2 += pow(a, j)*cosf(2.0f*PI_VAL*pow(b, j)*(0.5f));
			}
			candidate->obj_func_val = temp1 - options->dimensions*temp2;
			candidate->obj_func_val += SHIFTING_VALUE;
			if(options->debugging_level>=LOG_VERBOSE){
				LOGV("WEIERSTRASS_F-> Obj val: %4.4f", candidate->obj_func_val);
			}
			break;
		}
		default:
		{
			if(options->debugging_level>=LOG_WARNINGS){
				LOGW("This function is not supported: %d, exiting ...",
					options->function_to_use);
			}
			return WRONG_BENCHMARK_FUNCTION;
		}

	}
	return SUCCESS;
}

/*
	Estimate the average objective function value of the whole population
*/
uint8_t estimate_avrg_obj(Virus_Population *viruses){
	User_Options *options = viruses->usr_options;
	Virus *population = viruses->solutions;
	uint32_t n_viruses = viruses->n_viruses, i;
	float *average_obj = &(viruses->average_obj_func_value);

	*average_obj = 0.0f;
	for(i=0;i<n_viruses;i++){
		*average_obj += population[i].obj_func_val;
	}
	*average_obj /= (float)n_viruses;

	if(options->debugging_level >= LOG_DEBUG){
		LOGD("Average Obj Func Value in this replication is: %4.4f",
			viruses->average_obj_func_value);
	}

	return SUCCESS;
}

/*
	Determine which viruses are to be deleted, according to the
	average objective function value determined before. Mark all
	the viruses to be removed, turning ON the removal FLAG
*/
uint8_t population_contest(Virus_Population *viruses){
	User_Options *options = viruses->usr_options;
	Virus *population = viruses->solutions;
	uint32_t n_viruses = viruses->n_viruses;
	int i;
	float average_obj = viruses->average_obj_func_value;
	uint32_t count=0, n_remove = viruses->n_viruses_to_remove;
	uint32_t n_strong = viruses->n_strong;

	for(i=0;i<n_viruses;i++){
		if(count>=n_remove){
			break;
		}

		if(options->debugging_level>=LOG_INFO){
			LOGI("Virus Type: %d", population[i].virus_type);
		}

		if(population[i].obj_func_val>=average_obj && population[i].virus_type != STRONG_VIRUS){
			population[i].remove_virus = TRUE;
			if(options->debugging_level>=LOG_INFO){
				LOGI("Virus %d is going to be removed <%4.4f, %4.4f>", i,
					population[i].obj_func_val, average_obj);
			}
			count++;
		}else{
			if(options->debugging_level>=LOG_INFO){
				LOGI("Virus %d is NOT going to be removed <%4.4f, %4.4f>", i,
					population[i].obj_func_val, average_obj);
			}
			population[i].remove_virus = FALSE;
		}
	}

	if(count<n_remove){
		/*we need to remove more*/
		if(options->debugging_level>=LOG_INFO){
			LOGI("performing a second round selection, not enough memebers were selected...");
		}
		while(TRUE){
			i--;
			if(i<0){
				/*no more to choose from*/
				break;
			}

			/*remove in reverse*/
			if(!population[i].remove_virus && population[i].virus_type != STRONG_VIRUS){
				population[i].remove_virus = TRUE;
				if(options->debugging_level>=LOG_INFO){
					LOGI("Virus %d is going to be removed <%4.4f, %4.4f>", i,
						population[i].obj_func_val, average_obj);
				}
				count++;
			}else{
				if(options->debugging_level>=LOG_INFO){
					LOGI("Virus %d is NOT going to be removed <%4.4f, %4.4f>", i,
						population[i].obj_func_val, average_obj);
				}
			}
			if(count>=n_remove || count<=n_strong){
				/*we reached the required value or no more common ones*/
				if(count<=n_strong){
					if(options->debugging_level>=LOG_WARNINGS){
						LOGW("Not enough members in the population, only the strong members will survive");
					}
				}
				break;
			}
		}
	}

	viruses->remaining_after = n_viruses-count;

	if(options->debugging_level >= LOG_DEBUG){
		LOGD("%d viruses marked from removal in the population, %d are to be kept",
			count, viruses->remaining_after);
	}

	return SUCCESS;
}


/*slow copy function of the new viruses in the new population*/
uint8_t create_virus(Virus *new_solution, Virus *survivor, uint32_t dimensions){
	/*perform memory creation and copy of values*/
	new_solution->obj_func_val = survivor->obj_func_val;
	new_solution->remove_virus = survivor->remove_virus;
	new_solution->virus_type = survivor->virus_type;
	new_solution->variable_val=NULL;
	new_solution->variable_val = (float*)malloc(sizeof(float)*dimensions);
	if(!new_solution->variable_val){
		return CANT_CREATE_VIRUS;
	}

	/*
		slow copy, any better way?
	*/
	memcpy(new_solution->variable_val, survivor->variable_val, sizeof(float)*dimensions);

	/*free old vector of solutions*/
	free(survivor->variable_val);

	return SUCCESS;
}

void swap_solutions(Virus *not_better, Virus *better, float *temp_vars, uint32_t dimensions){
	/*exhange decision variables*/
	memcpy(temp_vars, not_better->variable_val, sizeof(float)*dimensions);
	memcpy(not_better->variable_val, better->variable_val, sizeof(float)*dimensions);
	memcpy(better->variable_val, temp_vars, sizeof(float)*dimensions);

	/*exchange normal fields*/
	float temp;
	temp = not_better->obj_func_val;
	not_better->obj_func_val = better->obj_func_val;
	better->obj_func_val = temp;

	uint8_t tt;
	tt = not_better->remove_virus;
	not_better->remove_virus = better->remove_virus;
	better->remove_virus = tt;

	tt = not_better->virus_type;
	not_better->virus_type = better->virus_type;
	better->virus_type = tt;
	better->virus_type = STRONG_VIRUS;
	not_better->virus_type = COMMON_VIRUS;
}

/*
	Remove the viruses flagged to be deleted from the pooled
	population
*/
uint8_t population_maintenance(Virus_Population *pooled_viruses){
	User_Options *options = pooled_viruses->usr_options;
	Virus *old_population = pooled_viruses->solutions;

	uint32_t remaining = pooled_viruses->remaining_after, i, j, k;
	Virus *new_population=NULL;
	uint32_t n_viruses = pooled_viruses->n_viruses;
	uint32_t n_strong = pooled_viruses->n_strong;

	float *temp_vars = (float*)malloc(sizeof(float)*options->dimensions);

	/*container for new population*/
	new_population = (Virus*)malloc(sizeof(Virus)*remaining);

	if(!new_population){
		if(options->debugging_level>=LOG_ERRORS){
			LOGE("Failed to create new population of viruses, Ran out of Memory...!!!");
		}
		return FAILED_TO_CREATED_NEW_POPULATION;
	}

	j=0;

	if(options->debugging_level >= LOG_DEBUG){
		LOGD("%d viruses have been labels as strongs", pooled_viruses->n_strong);
	}

	for(i=0; i<n_viruses; i++){
		if(!((old_population+i)->remove_virus)){
			/*perform copy of those surviving viruses*/
			if(create_virus(new_population+j, old_population+i, options->dimensions)!=SUCCESS){
				if(options->debugging_level>=LOG_ERRORS){
					LOGE("Failed to create new population of viruses, Ran out of Memory...!!!");
				}
				return FAILED_TO_CREATED_NEW_POPULATION;
			}
			/*
				Check if this virus is better than the current strong ones so we
				can keep the ranking viruses sorted at all times
			*/
			if(j>0){//only if we have more than 1 solution
				for(k=0; (k<n_strong && k<j); k++){//to the limit or until n_strong
					if((new_population+j)->obj_func_val < (new_population+k)->obj_func_val){
						/*this virus is better than the current strong at this location*/
						swap_solutions((new_population+k), (new_population+j), temp_vars,
							options->dimensions);
						break;//dont change anything anymore
					}
				}
			}
			j++;
		}else{
			/*just free the memory allocated from the variables*/
			free((old_population+i)->variable_val);
		}
	}

	/*destroy all the solutions in the old population*/
	free(old_population);

	/*make solution pointer to address new population*/
	pooled_viruses->solutions = new_population;
	pooled_viruses->n_viruses = remaining;
	pooled_viruses->remaining_after=0;
	pooled_viruses->n_viruses_to_remove=0;

	free(temp_vars);

	return SUCCESS;
}

/*Time query functions*/
void start_clock(){
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
}

float end_query_clock(){
	float elapsed;
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	elapsed = (float)ElapsedMicroseconds.QuadPart/(float)Frequency.QuadPart;

	return elapsed/1000000.0f;
}

void free_pop_viruses(Virus_Population *viruses){
	User_Options *options = viruses->usr_options;
	Virus *solutions = viruses->solutions;
	uint32_t pop_size = viruses->n_viruses;
	uint32_t i;

	if(options->debugging_level>=LOG_DEBUG){
		LOGD("Freeing all viruses in the population...");
	}

	for(i=0;i<pop_size;i++){
		if(options->debugging_level>=LOG_INFO){
			LOGI("Removing virus %d", i);
		}
		free(solutions[i].variable_val);
	}

	free(solutions);
}
