#pragma once
#ifndef _PLATFORM_H
#define _PLATFORM_H

/*-----------------------------------------------------------------------------
   Name:	Platform.h
   Desc:	Used to express platform and hardware differences
         For Instance:  Windows, UNIX, etc.
         Or GPU Hardware:  GTX 480, GTX 580, etc.

   Disclaimer:
      This software is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

   Log:   Created by Shawn D. Brown (7/14/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Platform Definitions
-------------------------------------*/

//-----
// OS Platform
//-----
#define OS_PLATFORM_UNIX       1
#define OS_PLATFORM_WIN_VS2008 2

//#define OS_PLATFORM OS_PLATFORM_UNIX
#define OS_PLATFORM OS_PLATFORM_WIN_VS2008


//-----
// CPU Platform (Intel or ???)
//-----

#define CPU_UNKNOWN 0
#define CPU_INTEL_X86     86
#define CPU_INTEL_X64     64

//#define CPU_PLATFORM CPU_UNKNOWN
//#define CPU_PLATFORM CPU_INTEL_X86
#define CPU_PLATFORM CPU_INTEL_X64


//-----
// GPU Platform (Hardware)
//-----

#define GPU_UNKNOWN     0
#define GPU_GTX_285     285
#define GPU_GTX_480     480
#define GPU_GTX_560_TI  561
#define GPU_GTX_560_M   562
#define GPU_GTX_580     580
#define GPU_TELSA_M2050 2050


#define GPU_PLATFORM GPU_GTX_480
//#define GPU_PLATFORM GPU_GTX_560_M
//#define GPU_PLATFORM GPU_GTX_580


//-----
// # of SM's on GPU card
//-----

#if GPU_GTX_470 == GPU_PLATFORM
   #define GPU_NUM_SM 14
#elif GPU_GTX_480 == GPU_PLATFORM
   #define GPU_NUM_SM 15
#elif GPU_GTX_560_TI == GPU_PLATFORM
   #define GPU_NUM_SM 12
#elif GPU_GTX_560_M == GPU_PLATFORM
   #define GPU_NUM_SM 6
#elif GPU_GTX_580 == GPU_PLATFORM
   #define GPU_NUM_SM 16
#elif GPU_TELSA_M2050 == GPU_PLATFORM
   #define GPU_NUM_SM 14
#endif



#endif // _PLATFORM_H

