#pragma once
#ifndef _TRISH_TRAITS_H
#define _TRISH_TRAITS_H

/*-----------------------------------------------------------------------------
   Name:	TRISH_Traits.h
   Desc:	TRISH trait type definitions go here

   Disclaimer:
      This software is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

   Log:   Created by Shawn D. Brown (4/24/12)
-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
  Includes
-----------------------------------------------------------------------------*/

#include "Platform.h"
#include "BaseDefs.h"


/*-----------------------------------------------------------------------------
  Templates
-----------------------------------------------------------------------------*/

//-----
// Useful Traits for TRISH types
//-----

template < typename valT >
struct TRISH_traits 
{
   // Deliberately empty
};


//-----
// Define Useful TRISH traits using partial specialization
//-----

template <>
struct TRISH_traits<I8>
{
   typedef I8   base_type;       // Base Type      
   typedef I32  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F32  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<U8>
{
   typedef U8   base_type;       // Base Type      
   typedef U32  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F32  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<I16>
{
   typedef I16  base_type;       // Base Type      
   typedef I32  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F32  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<U16>
{
   typedef U16  base_type;       // Base Type      
   typedef U32  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F32  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<I32>
{
   typedef I32  base_type;       // Base Type      
   typedef I32  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F64  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<U32>
{
   typedef U32  base_type;       // Base Type      
   typedef U32  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F64  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<I64>
{
   typedef I64  base_type;       // Base Type      
   typedef I64  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F64  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<U64>
{
   typedef U64  base_type;       // Base Type      
   typedef U64  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F64  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<F32>
{
   typedef F32  base_type;       // Base Type      
   typedef F32  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F64  convert_type;    // Default conversion type
};

template <>
struct TRISH_traits<F64>
{
   typedef F64  base_type;       // Base Type      
   typedef F64  upscale_type;    // Intermediate type (for faster processing)
   typedef U32  bin_type;        // Bin storage type
   typedef F64  convert_type;    // Default conversion type
};


#endif	// _TRISH_TRAITS_H

