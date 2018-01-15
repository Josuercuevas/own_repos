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

#ifndef _PROTOTYPES_HEADER_
#define _PROTOTYPES_HEADER_

#pragma once //includes just once

/* For general information */
#include "LICENSE.h"

/* General Includes */
#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <math.h>
#include <time.h>


/*	supports 16 foreground and background colors:
	[0-15]=[black,navy,green,teal,maroon,purple,olive,silver,gray,blue,lime,aqua,red,fuchsia,yellow,white]
	using the <color>{your text here} syntax. for example:
	red{text printed in red-on-black color}
	redwhite{text printed in red-on-white color}
*/
#define CFPRINTF_MAX_BUFFER_SIZE (8192)
int cfprintf(FILE *f, const char *fmt, ...); // exactly as fprintf but with color support


/*
	Predefined structures, check their fields
	at below
*/
typedef struct _user_options User_Options;
typedef struct _virus Virus;
typedef struct _Virus_Population Virus_Population;

/*
	General types that can be used
	to reduce computation and memory usage
*/
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

/*
	Logging printf functions that can be used
*/
#define LOGW(fmt, ...) cfprintf(stderr, "yellowblack{WARNING: %s %d: " fmt "}\n", \
	__FUNCTION__, __LINE__, __VA_ARGS__)

#define LOGI(fmt, ...) cfprintf(stderr, "aquablack{INFO: %s %d: " fmt "}\n", \
	__FUNCTION__, __LINE__, __VA_ARGS__)

#define LOGE(fmt, ...) cfprintf(stderr, "redblack{ERROR: %s %d: " fmt "}\n", \
	__FUNCTION__, __LINE__, __VA_ARGS__)

#define LOGD(fmt, ...) cfprintf(stderr, "limeblack{DEBUG: %s %d: " fmt "}\n", \
	__FUNCTION__, __LINE__, __VA_ARGS__)

#define LOGV(fmt, ...) cfprintf(stderr, "whiteblack{VERBOSE: %s %d: " fmt "}\n", \
	__FUNCTION__, __LINE__, __VA_ARGS__)


/*Dimensionality of the problem to be solved*/
static uint32_t DIM=10;
static float SHIFTING_VALUE=0.0f;
static uint32_t INIT_POP = 10;
static uint32_t N_STRONG_INIT = 2;
static uint8_t R_STRONG = 2;
static uint8_t R_COMMON = 1;

#define MAX_VIRUSES (500)

/*average objective function value to be use for virus removal*/
static float AVERAGE_OBJ_FUN_VALUE=99999999.99f;

/*
	Benchmark functions that can be selected
	0-bit stride
*/
enum _benchmark_functions{
	SPHERE_F=0,
	ROSENBROCK_F,
	ACKLEY_F,
	SCHWEFEL22_F,
	SCHWEFEL26_F,
	RASTRIGIN_F,
	GRIEWANK_F,
	WEIERSTRASS_F
};

/*Global minimum of each function above*/
#define SPHERE_GM (0.0f)
#define ROSENBROCK_GM (0.0f)
#define ACKLEY_GM (0.0f)
#define SCHWEFEL22_GM (0.0f)
#define RASTRIGIN_GM (0.0f)
#define GRIEWANK_GM (0.0f)
#define WEIERSTRASS_GM (0.0f)
/*this one needs to be reset, since is dimension dependent*/
static float SCHWEFEL26_GM = DIM*(-418.9829f);

/*Benchmark function, range values [min, max]*/
static float SPHERE_RANGE[2]={-100.0f, 100.0f};
static float ROSENBROCK_RANGE[2]={-30.0f, 30.0f};
static float ACKLEY_RANGE[2]={-32.0f, 32.0f};
static float SCHWEFEL22_RANGE[2]={-10.0f, 10.0f};
static float SCHWEFEL26_RANGE[2]={-500.0f, 500.0f};
static float RASTRIGIN_RANGE[2]={-5.12f, 5.12f};
static float GRIEWANK_RANGE[2]={-600.0f, 600.0f};
static float WEIERSTRASS_RANGE[2]={-0.50f, 0.50f};

#define INF_VAL (999999999999.99f)

/*
	Debug levels that can be chosen by the user to run this program
	31-bit stride
*/
enum _log_levels{
	LOG_NONE=32,
	LOG_ERRORS,
	LOG_WARNINGS,
	LOG_DEBUG,
	LOG_INFO,
	LOG_VERBOSE,
};
static uint8_t DEBUG_LEVEL=LOG_VERBOSE;

/*
	Error codes to be used to signal any problem
	during the execution of the program 63-bit stride
*/
enum _error_codes{
	SUCCESS=64,
	USER_OPTIONS_ERROR,
	DIMENSIONS_TOO_LARGE,
	WRONG_BENCHMARK_FUNCTION,
	FAILED_TO_CREATED_NEW_POPULATION,
	CANT_CREATE_VIRUS,
	FAILED_TO_CREATED_INITIAL_POPULATION,

	UNKNOWN
};


/*
	Main trick of the virus reproduction, these are the
	corresponding flags for later usage, 95-bit stride
*/
enum _virus_type{
	STRONG_VIRUS=96,
	COMMON_VIRUS
};





/*
	Main structure to be used when the user enters any option
	otherwise predefined values are to be set during the
	initialization process
*/
struct _user_options{
	float shifting_value;
	uint32_t dimensions;
	uint32_t initial_population;
	uint8_t bench_function;
	uint8_t debugging_level;
	uint8_t function_to_use;

	/*
	memory alignment for different machines, so we dont
	have problems when compiling and running in different
	machines
		WARNING:
			DO NOT CHANGE THIS, unless u know what u are doing!!
	*/
	uint8_t _align[22];//bit alignment
};

/*
	Virus or candidate solution to be used to
	store all the necesary information
*/
struct _virus{
	/*decision variables values*/
	float *variable_val;
	/*objective function value*/
	float obj_func_val;
	/*type of virus*/
	uint8_t virus_type;
	/*flag to determine if we need to remove it or not*/
	uint8_t remove_virus;

	/*
	memory alignment for different machines, so we dont
	have problems when compiling and running in different
	machines
		WARNING:
			DO NOT CHANGE THIS, unless u know what u are doing!!
	*/
	uint8_t _align[25];//bit alignment
};

/* Virus/Solution structure to be used through the whole program */
struct _Virus_Population{
	/*average objective function value*/
	float average_obj_func_value;
	/*Number of viruses in the population*/
	uint32_t n_viruses;
	/*intensity value for the perturbation modification*/
	uint32_t intensity;
	/*growing rate for strong viruses*/
	uint8_t growing_rate_strong;
	/*growing rate for common viruses*/
	uint8_t growing_rate_common;
	/*How many viruses are to be removed, generated by user here*/
	uint32_t n_viruses_to_remove;
	/*number of strong viruses in population given by user*/
	uint32_t n_strong;
	/*number of viruses to keep after maintenance*/
	uint32_t remaining_after;

	/*Solutions containing the variables values*/
	Virus *solutions;
	/*options inputed by the user*/
	User_Options *usr_options;

	/*
		Memory alignment variable
	*/
	uint8_t _align[32];
};



/*Functions to be used for this program*/
void reset_functions(User_Options *options);
void swap_solutions(Virus *not_better, Virus *better,
					float *temp_vars, uint32_t dimensions);

/*Population generation functions*/
uint8_t create_virus(Virus *new_solution, Virus *survivor, uint32_t dimensions);
uint8_t create_first_population(Virus_Population *viruses);
uint8_t generate_new_solutions(Virus_Population *current_viruses,
							   Virus_Population *new_viruses);
uint8_t generate_pooled_population(Virus_Population *pooled_viruses,
							   Virus_Population *new_viruses);

void free_pop_viruses(Virus_Population *viruses);


/* Fitting related functions*/
uint8_t estimate_ObjFunc(Virus *candidate, User_Options *options);
uint8_t estimate_avrg_obj(Virus_Population *viruses);

/*Population maintenance related function*/
uint8_t population_contest(Virus_Population *viruses);
uint8_t population_maintenance(Virus_Population *pooled_viruses);

/*Computation time query functions, with a precision in Microseconds*/
static LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
static LARGE_INTEGER Frequency;
void start_clock();
float end_query_clock();

#endif //_PROTOTYPES_HEADER_
