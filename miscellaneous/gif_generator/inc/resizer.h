/*
 * resizer.h
 *
 *  Created on: Oct 6, 2015
 *      Author: josue

 	 Copyright (C) <2018>  <Josue R. Cuevas>

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

#ifndef RESIZER_H_
#define RESIZER_H_

#include <stdio.h>   // for FILE*
#include <string.h>  // for memcpy and bzero
#include <stdint.h>  // for integer typedefs

void resize(uint8_t *src, uint8_t *dst, int src_w, int src_h, int dst_w,
		int dst_h, const char *pFilter);
void resample(uint8_t *src, uint8_t *dst, int src_w, int src_h, int dst_w,
		int dst_h, const char *pFilter);



#endif /* RESIZER_H_ */
