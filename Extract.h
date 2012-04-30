#ifndef _EXTRACT_H
#define _EXTRACT_H
/*-----------------------------------------------------------------------------
   Name: Extract.h
   Desc: Helper templates to extract bytes/words from DWORDS

   Notes:

   Disclaimer:
      This software is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

   Log:   Created by Shawn D. Brown (4/25/12)
-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
  Includes
-----------------------------------------------------------------------------*/

// Local Includes
#include "Platform.h"
#include "BaseDefs.h"
#include "TRISH_traits.h"


/*-----------------------------------------------------------------------------
  Extractor Templates
-----------------------------------------------------------------------------*/

/*-----------------------------------------------
  Name:  ExtractBytes4
  Desc:  Extracts four 8-bit bytes 
         from a singleton 32-bit
         unsigned integer (U32)
-----------------------------------------------*/

template < typename valT >  // Output Type
struct ExtractBytes
{
   static void 
   __host__ __device__ __forceinline__
   Extract4
   ( 
      valT &  out1,    // OUT - extracted bytes
      valT &  out2,   
      valT &  out3,
      valT &  out4,
      U32     in14     // IN - 32-bit value to extract 4 bytes from
   ) 
   {
      // Do nothing on purpose !!!
      // Specialize this place holder template
      // using partial specialization
   }
};


// ExtractBytes<U32> partial specialization
// 32-bit unsigned integer version
template <>
struct ExtractBytes<U32>
{
   static void 
   __host__ __device__ __forceinline__
   Extract4
   ( 
      U32 &  out1,    // OUT - extracted bytes (stored as U32's)
      U32 &  out2,   
      U32 &  out3,
      U32 &  out4,
      U32    in14     // IN - 32-bit value to extract 4 bytes from
   ) 
   {
      // Extract singleton 'Bytes' from 32-bit storage value
      out1 = in14 >>  0u;
      out2 = in14 >>  8u;
      out3 = in14 >> 16u;
      out4 = in14 >> 24u;

      // Mask off unwanted bits
      out1 = out1 & 0xFFu;
      out2 = out2 & 0xFFu;
      out3 = out3 & 0xFFu;
      out4 = out4 & 0xFFu;
   }
};


// ExtractBytes<I32> partial specialization
// 32-bit (signed) integer version
template <>
struct ExtractBytes<I32>
{
   static void 
   __host__ __device__ __forceinline__
   Extract4
   ( 
      I32 &  out1,    // OUT - extracted bytes (stored as I32's)
      I32 &  out2,   
      I32 &  out3,
      I32 &  out4,
      U32    in14     // IN - 32-bit value to extract 4 bytes from
   ) 
   {
      // Extract singleton 'Bytes' from 32-bit storage value
      U32 b1, b2, b3, b4;
      b1 = in14 >>  0u;
      b2 = in14 >>  8u;
      b3 = in14 >> 16u;
      b4 = in14 >> 24u;

      b1 = b1 & 0xFFu;
      b2 = b2 & 0xFFu;
      b3 = b3 & 0xFFu;
      b4 = b4 & 0xFFu;


      //-----
      // Sign-extend 8-bit values into 32-bit values
      //-----
      
      // Move 8-bit sign bit into 32-bit sign position
      b1 <<= 24u;
      b2 <<= 24u;
      b3 <<= 24u;
      b4 <<= 24u;

      // Convert to signed result
      out1 = (I32)b1;
      out2 = (I32)b2;
      out3 = (I32)b3;
      out4 = (I32)b4;

      // Shift back to correct position (with sign-extension side-effect)
      out1 >>= 24u;
      out2 >>= 24u;
      out3 >>= 24u;
      out4 >>= 24u;
   }
};



/*-----------------------------------------------
  Name:  ExtractWords4
  Desc:  Extracts four 16-bit words from 
         two 32-bit unsigned integers (U32)
-----------------------------------------------*/

template < typename valT >  // Output Type
struct ExtractWords
{
   static void 
   __host__ __device__ __forceinline__
   Extract2
   ( 
      valT &  out1,    // OUT - extracted words
      valT &  out2,   
      U32     in12     // IN  - single 32-bit value to extract 2 words from
   ) 
   {
      // Do nothing on purpose !!!
      // specialize this placeholder template
      // using partial specialization
   }

   static void 
   __host__ __device__ __forceinline__
   Extract4
   ( 
      valT &  out1,    // OUT - extracted words
      valT &  out2,   
      valT &  out3,
      valT &  out4,
      U32     in12,    // IN - two 32-bit values to extract 4 words from
      U32     in34     //      ditto
   ) 
   {
      // Do nothing on purpose !!!
      // specialize this placeholder template
      // using partial specialization
   }
};


// ExtractWords<U32> partial specialization
// 32-bit unsigned integer version
template <>
struct ExtractWords<U32>
{
   static void 
   __host__ __device__ __forceinline__
   Extract2
   ( 
      U32 &  out1,    // OUT - extracted words (stored as U32's)
      U32 &  out2,   
      U32    in12     // IN - single 32-bit value to extract 2 words from
   ) 
   {
      // Extract two 'Words' from 32-bit storage value
      out1 = in12 & 0xFFFFu;
      out2 = in12 >> 16u;
   }

   static void 
   __host__ __device__ __forceinline__
   Extract4
   ( 
      U32 &  out1,    // OUT - extracted words (stored as U32's)
      U32 &  out2,   
      U32 &  out3,
      U32 &  out4,
      U32    in12,    // IN - two 32-bit values to extract 4 words from
      U32    in34     //      ditto
   ) 
   {
      // Extract four 'Words' from two 32-bit storage values
      out1 = in12 & 0xFFFFu;
      out2 = in12 >> 16u;
      out3 = in34 & 0xFFFFu;
      out4 = in34 >> 16u;
   }
};


// ExtractWords<I32> partial specialization
// 32-bit (signed) integer version
template <>
struct ExtractWords<I32>
{
   static void 
   __host__ __device__ __forceinline__
   Extract2
   ( 
      I32 &  out1,    // OUT - extracted words (stored as I32's)
      I32 &  out2,   
      U32    in12     // IN - single 32-bit value to extract 2 words from
   ) 
   {
      //-----
      // Extract two 'Words' from a single 32-bit storage value
      //-----

      U32 b1, b2;
      b1 = in12 & 0xFFFFu;
      b2 = in12 >> 16u;


      //-----
      // Sign-extend 16-bit values into 32-bit storage values
      //-----
      
      // Move 16-bit sign bit into 32-bit sign position
      b1 <<= 16u;
      b2 <<= 16u;

      // Convert to signed result
      out1 = (I32)b1;
      out2 = (I32)b2;

      // Shift back to correct position (with sign-extend side-effect)
      out1 >>= 16u;
      out2 >>= 16u;
   }

   static void 
   __host__ __device__ __forceinline__
   Extract4
   ( 
      I32 &  out1,    // OUT - extracted words (stored as I32's)
      I32 &  out2,   
      I32 &  out3,
      I32 &  out4,
      U32    in12,    // IN - two 32-bit values to extract 4 words from
      U32    in34     //      ditto
   ) 
   {
      //-----
      // Extract four 'Words' from two 32-bit storage values
      //-----

      U32 b1, b2, b3, b4;
      b1 = in12 & 0xFFFFu;
      b2 = in12 >> 16u;
      b3 = in34 & 0xFFFFu;
      b4 = in34 >> 16u;


      //-----
      // Sign-extend 16-bit values into 32-bit storage values
      //-----
      
      // Move 16-bit sign bit into 32-bit sign position
      b1 <<= 16u;
      b2 <<= 16u;
      b3 <<= 16u;
      b4 <<= 16u;

      // Convert to signed result
      out1 = (I32)b1;
      out2 = (I32)b2;
      out3 = (I32)b3;
      out4 = (I32)b4;

      // Shift back to correct position (with sign-extend side-effect)
      out1 >>= 16u;
      out2 >>= 16u;
      out3 >>= 16u;
      out4 >>= 16u;
   }
};


#endif // _EXTRACT_H