#ifndef POITRI_H_
#define POITRI_H_

#include "vec.h"

// find distance x0 is from segment x1-x2
float point_segment_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, Vec3f &r)
{
   Vec3f dx(x2-x1);
   double m2=mag2(dx);
   // find parameter value of closest point on segment
   float s12=(float)(dot(x2-x0, dx)/m2);
   if(s12<0){
      s12=0;
   }else if(s12>1){
      s12=1;
   }
   // and find the distance
   r = s12*x1+(1-s12)*x2;
   return dist(x0, s12*x1+(1-s12)*x2);
}

// find distance x0 is from triangle x1-x2-x3
float point_triangle_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, const Vec3f &x3, Vec3f &r)
{
   // first find barycentric coordinates of closest point on infinite plane
   Vec3f x13(x1-x3), x23(x2-x3), x03(x0-x3);
   float m13=mag2(x13), m23=mag2(x23), d=dot(x13,x23);
   float invdet=1.f/max(m13*m23-d*d,1e-30f);
   float a=dot(x13,x03), b=dot(x23,x03);
   // the barycentric coordinates themselves
   float w23=invdet*(m23*a-d*b);
   float w31=invdet*(m13*b-d*a);
   float w12=1-w23-w31;
   if(w23>=0 && w31>=0 && w12>=0){ // if we're inside the triangle
      r = w23*x1+w31*x2+w12*x3;
      return dist(x0, w23*x1+w31*x2+w12*x3);
   }else{ // we have to clamp to one of the edges
      Vec3f r1(0);
      Vec3f r2(0);

      if(w23>0){ // this rules out edge 2-3 for us
         //return min(point_segment_distance(x0,x1,x2), point_segment_distance(x0,x1,x3,r));
         float d1 = point_segment_distance(x0,x1,x2,r1);
         float d2 = point_segment_distance(x0,x1,x3,r2);

         if(d1<d2){
           r = r1;
           return d1;
         }else{
           r = r2;
           return d2;
         }
      }else if(w31>0){ // this rules out edge 1-3
         //return min(point_segment_distance(x0,x1,x2), point_segment_distance(x0,x2,x3));
         float d1 = point_segment_distance(x0,x1,x2,r1);
         float d2 = point_segment_distance(x0,x2,x3,r2);

         if(d1<d2){
           r = r1;
           return d1;
         }else{
           r = r2;
           return d2;
         }
      }else{ // w12 must be >0, ruling out edge 1-2
         //return min(point_segment_distance(x0,x1,x3), point_segment_distance(x0,x2,x3));
         float d1 = point_segment_distance(x0,x1,x3,r1);
         float d2 = point_segment_distance(x0,x2,x3,r2);

         if(d1<d2){
           r = r1;
           return d1;
         }else{
           r = r2;
           return d2;
         }
      }
   }
}

#endif