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

#ifdef _WIN32
#include <windows.h>
static HANDLE hConsole = 0;
#define BACK_TO_DEFAULT (15)
#else
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#define BACK_TO_DEFAULT		"\33[0m"
//                     black  blue   lgtgrn green  red    purple   olive       gray   silver blue   lgtgrn blue   red    purple yellow white
const char *FGs[]   = {"1m" , "34m", "32m", "32m", "31m", "35m",   "36m"    ,  "37m", "37m", "34m", "32m", "34m", "31m", "35m", "33m", "m"};
//                     black  blue   lgtgrn green  red    purple   olive       gray   silver blue   lgtgrn blue   red    purple yellow white
const char *BGs[]   = {"40m", "44m", "42m", "46m", "41m", "45m",   "46m"    ,  "47m", "47m", "44m", "46m", "44m", "41m", "45m", "43m", "m"};
#endif

void cfprintf_init() {
#ifdef _WIN32
	if (hConsole)
		return;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#else
#endif
}
int qsort_positions(const void *x, const void *y) {
	// Returns -1 if x < y
    //          0 if x == y
    //         +1 if x > y
	size_t *xx = (size_t*)x;
	size_t *yy = (size_t*)y;
	if (*xx<*yy) return -1;
	else if (*xx>*yy) return 1;
	else return 0;
}
int cfprintf(FILE *f, const char *fmt, ...) {
	struct POS {
		size_t start, finish;
		int    color;
	};
	size_t		pos_counter;
	struct POS	pos[128];
	char		*pos_0, *pos_1, *pos_2;
	char		*sptr;
	va_list		argp;
	int         c;
	size_t      cc;
	const char	*clrs[] = {
		// 16 Colors for each of the following Backgrounds:
		// BLACK        BLUE			GREEN			DEEP GREEN		DARK RED		PURPLE			OLIVE			GRAY			GRAY			BLUE		    LIGHT GREEN      LGT BLUE    RED       PURPLE          YELLOW        WHITE
		"blackblack"	,"navyblack"	,"greenblack"	,"tealblack"	,"maroonblack"	,"purpleblack"	,"oliveblack"	,"silverblack"	,"grayblack"	,"blueblack"	,"limeblack"	,"aquablack","redblack","fuchsiablack","yellowblack","whiteblack",
		"blacknavy"		,"navynavy"		,"greennavy"	,"tealnavy"		,"maroonnavy"	,"purplenavy"	,"olivenavy"	,"silvernavy"	,"graynavy"		,"bluenavy"		,"limenavy"		,"aquanavy","rednavy","fuchsianavy","yellownavy","whitenavy",
		"blackgreen"	,"navygreen"	,"greengreen"	,"tealgreen"	,"maroongreen"	,"purplegreen"	,"olivegreen"	,"silvergreen"	,"graygreen"	,"bluegreen"	,"limegreen"	,"aquagreen","redgreen","fuchsiagreen","yellowgreen","whitegreen",
		"blackteal"		,"navyteal"		,"greenteal"	,"tealteal"		,"maroonteal"	,"purpleteal"	,"oliveteal"	,"silverteal"	,"grayteal"		,"blueteal"		,"limeteal"		,"aquateal","redteal","fuchsiateal","yellowteal","whiteteal",
		"blackmaroon"	,"navymaroon"	,"greenmaroon"	,"tealmaroon"	,"maroonmaroon"	,"purplemaroon"	,"olivemaroon"	,"silvermaroon"	,"graymaroon"	,"bluemaroon"	,"limemaroon"	,"aquamaroon","redmaroon","fuchsiamaroon","yellowmaroon","whitemaroon",
		"blackpurple"	,"navypurple"	,"greenpurple"	,"tealpurple"	,"maroonpurple"	,"purplepurple"	,"olivepurple"	,"silverpurple"	,"graypurple"	,"bluepurple"	,"limepurple"	,"aquapurple","redpurple","fuchsiapurple","yellowpurple","whitepurple",
		"blackolive"	,"navyolive"	,"greenolive"	,"tealolive"	,"maroonolive"	,"purpleolive"	,"oliveolive"	,"silverolive"	,"grayolive"	,"blueolive"	,"limeolive"	,"aquaolive","redolive","fuchsiaolive","yellowolive","whiteolive",
		"blacksilver"	,"navysilver"	,"greensilver"	,"tealsilver"	,"maroonsilver"	,"purplesilver"	,"olivesilver"	,"silversilver"	,"graysilver"	,"bluesilver"	,"limesilver"	,"aquasilver","redsilver","fuchsiasilver","yellowsilver","whitesilver",
		"blackgray"		,"navygray"		,"greengray"	,"tealgray"		,"maroongray"	,"purplegray"	,"olivegray"	,"silvergray"	,"graygray"		,"bluegray"		,"limegray"		,"aquagray","redgray","fuchsiagray","yellowgray","whitegray",
		"blackblue"		,"navyblue"		,"greenblue"	,"tealblue"		,"maroonblue"	,"purpleblue"	,"oliveblue"	,"silverblue"	,"grayblue"		,"blueblue"		,"limeblue"		,"aquablue","redblue","fuchsiablue","yellowblue","whiteblue",
		"blacklime"		,"navylime"		,"greenlime"	,"teallime"		,"maroonlime"	,"purplelime"	,"olivelime"	,"silverlime"	,"graylime"		,"bluelime"		,"limelime"		,"aqualime","redlime","fuchsialime","yellowlime","whitelime",
		"blackaqua"		,"navyaqua"		,"greenaqua"	,"tealaqua"		,"maroonaqua"	,"purpleaqua"	,"oliveaqua"	,"silveraqua"	,"grayaqua"		,"blueaqua"		,"limeaqua"		,"aquaaqua","redaqua","fuchsiaaqua","yellowaqua","whiteaqua",
		"blackred"		,"navyred"		,"greenred"		,"tealred"		,"maroonred"	,"purplered"	,"olivered"		,"silverred"	,"grayred"		,"bluered"		,"limered"		,"aquared","redred","fuchsiared","yellowred","whitered",
		"blackfuchsia"	,"navyfuchsia"	,"greenfuchsia"	,"tealfuchsia"	,"maroonfuchsia","purplefuchsia","olivefuchsia"	,"silverfuchsia","grayfuchsia"	,"bluefuchsia"	,"limefuchsia"	,"aquafuchsia","redfuchsia","fuchsiafuchsia","yellowfuchsia","whitefuchsia",
		"blackyellow"	,"navyyellow"	,"greenyellow"	,"tealyellow"	,"maroonyellow"	,"purpleyellow"	,"oliveyellow"	,"silveryellow"	,"grayyellow"	,"blueyellow"	,"limeyellow	","aquayellow","redyellow","fuchsiayellow","yellowyellow","whiteyellow",
		"blackwhite"	,"navywhite"	,"greenwhite"	,"tealwhite"	,"maroonwhite"	,"purplewhite"	,"olivewhite"	,"silverwhite"	,"graywhite"	,"bluewhite"	,"limewhite"	,"aquawhite","redwhite","fuchsiawhite","yellowwhite","whitewhite"
	};
	char        fmt_copy[CFPRINTF_MAX_BUFFER_SIZE];
	size_t      pc;
	size_t      remove_start, remove_finish;

	cfprintf_init();
	va_start(argp, fmt);
	vsnprintf_s(fmt_copy, CFPRINTF_MAX_BUFFER_SIZE, fmt, argp);
	va_end(argp);

	pos_counter=0;
	sptr = fmt_copy;
	pos_0 = (char*)sptr;
	while (pos_0) {
		for (c=0; c<256; ++c) {
			pos_0 = strstr(sptr, clrs[c]);
			if (pos_0) {
				pos_1 = strstr(pos_0, "{");
				if (pos_1 && (pos_1-pos_0)==strlen(clrs[c])) {
					pos_2 = strstr(pos_1, "}");
					if (pos_2) {
						pos[pos_counter].color  = c;
						pos[pos_counter].start  = pos_1+1-fmt_copy;
						pos[pos_counter].finish = pos_2-1-fmt_copy;
						++pos_counter;
						// clear string meta-color directive
						for (cc=0; cc<strlen(clrs[c])+1; ++cc)
							pos_0[cc]='#';
						*pos_2=' ';
						break;
					}
					else {
						cfprintf(stderr, "Warning: cfprintf identified a starting color label named yellowblack{%s} with no closing } bracket."
							"\nformat text:%s"
							"\nfull   text:%s"
							"\nposition in full text:redblack{%d}"
							"\n",
							clrs[c], fmt, fmt_copy, pos_0-fmt_copy);
					}
				}
			}
		}
	}
	qsort(pos, pos_counter, sizeof(struct POS), &qsort_positions);
	if (0==pos_counter) {
		fprintf(stderr, "%s", fmt_copy);
	}
	else {
		sptr=fmt_copy;
		remove_start = 0;
#ifdef _WIN32
		SetConsoleTextAttribute(hConsole, BACK_TO_DEFAULT);
#else
		fprintf(stderr, "%s", BACK_TO_DEFAULT);
#endif
		for (pc=0; pc<pos_counter; ++pc) {
			remove_start = pos[pc].start-1-strlen(clrs[pos[pc].color]);// inclusive
			if (pc>0)
				remove_start -= (pos[pc-1].finish+1+1);
			remove_finish = remove_start + strlen(clrs[pos[pc].color]); // inclusive
			// print from last pos until remove_start (not inclusive)
			sptr[remove_start]=0x0;
			fprintf(stderr, "%s", sptr);
			sptr[remove_start]='#';
#ifdef _WIN32
			SetConsoleTextAttribute(hConsole, pos[pc].color);
#else
			int bg = pos[pc].color / 16;
			int fg = pos[pc].color % 16;
			//fprintf(stderr, "\033[5m"); // blink
			//fprintf(stderr, "\033[25m"); // remove blink
			fprintf(stderr, "\033[1m"); // dim intensity. 1=bold
			fprintf(stderr, "\033[%s", BGs[bg]);
			fprintf(stderr, "\033[%s", FGs[fg]);
#endif
			sptr=fmt_copy+pos[pc].start;
			sptr[pos[pc].finish-pos[pc].start+1]=0x0;
			fprintf(stderr, "%s", sptr);
			sptr[pos[pc].finish-pos[pc].start+1]='#';
#ifdef _WIN32
			SetConsoleTextAttribute(hConsole, BACK_TO_DEFAULT);
#else
			fprintf(stderr, "%s", BACK_TO_DEFAULT);
#endif
			sptr=fmt_copy+pos[pc].finish+1+1; // jump over ending '}'
		}
		if (sptr)
			fprintf(stderr, "%s", sptr);
	}
	return 0;
}
