#pragma once
#ifndef ENHANCED_MADGWICK_HPP
#define ENHANCED_MADGWICK_HPP

#include <Eigen/Geometry>

/** This enhanced version of the Madgwick filter is a combination of ideas taken from Madgwick et al [2] 
 *	and Admiraal et al [1]. Admiraal improves upon the original Madgwick filter by deriving a steepest 
 *	(as apposed to gradient) descent formulation for calculating the quaternion estimate update 
 *	direction. This eliminates some of the numerical errors in Madgwick's version while also improving speed
 *	and precision. 
 *
 *	The notation used for documentation is as follows: [x](y)
 *	Where 'x' is the source and 'y' is the equation number
 *	
 *	[1] 	M. Admiraal, S. Wilson and R. Vaidyanathan, "Improved Formulation of the IMU and MARG 
 *			Orientation Gradient Descent Algorithm for Motion Tracking in Human-Machine Interfaces," in 
 *			International Conference on Multisensor Fusion and Integration for Intelligent Systems, Daegu, 
 *			2017.
 *
 *	[2] 	S. O. Madgwick, A. J. Harrison and R. Vaidyanathan, "Estimation of IMU and MARG Orientation
 *			Using A Gradient Descent Algorithm," in International Conference on Rehabilitation Robotics, 
 *			Zurich, 2011.
 *	
 **/
template<typename T>
class EnhancedMadgwick
{
public:

	EnhancedMadgwick()
	{
		qAccel.normalize();
		qMag.normalize();
	}
	~EnhancedMadgwick() = default;

private:
	Eigen::Quaternion<T> qAccel;
	Eigen::Quaternion<T> qMag;


	/** Calculates the gradient of the error function when referenced to gravity [1](35). This is then
	 *	used in the gradient descent algorithm to take a single step. This calculation assumes the 
	 *	acceleration reference vector is [0, 0, -1] because, ya know, gravity.
	 * 
	 *	@param[in]	estAccelQuaternion	The current quaternion estimate of orientation from accelerometer data
	 *	@param[in]	measuredAccelVector	A column matrix [x,y,z]' with measured accelerometer data
	 *	@return A quaternion representing the estimated orientation gradient
	 **/
	Eigen::Quaternion<T> accelGradient(Eigen::Quaternion<T>& estAccelQuaternion, Eigen::Matrix<T, 3, 1> measuredAccelVector)
	{
		Eigen::Matrix<T, 4, 1> result;
		Eigen::Matrix<T, 4, 3> leftProduct;
		Eigen::Matrix<T, 3, 1> rightProduct;

		T qw = estAccelQuaternion.w();
		T qx = estAccelQuaternion.x();
		T qy = estAccelQuaternion.y();
		T qz = estAccelQuaternion.z();
		T vmx = measuredAccelVector[0];
		T vmy = measuredAccelVector[1];
		T vmz = measuredAccelVector[2]; 


		leftProduct << 2 * (
			 qy, -qx, -qw,
		    -qz, -qw,  qx,
			 qw, -qz,  qy,
			-qx, -qy, -qz
			);

		rightProduct <<
			 2 * qw*qy, -2 * qx*qz, -vmx,
			-2 * qw*qx - 2 * qy*qz - vmy,
			-qw * qw + qx * qx + qy * qy - qz * qz - vmx;
			
		result = leftProduct * rightProduct;
		return Eigen::Quaternion<T>(result[0], result[1], result[2], result[3]);
	}

	/** Calculates the gradient of the error function when referenced to the earth's magnetic field. This 
	 *	is under the assumption that the field is perfectly planar in local space and can be represented
	 *	by the vector [Vrx, 0, Vrz]. [1](36)
	 *
	 **/
	Eigen::Quaternion<T> magGradient(Eigen::Quaternion<T>& estMagQuaternion, Eigen::Matrix<T, 3, 1> measuredMagVector, Eigen::Matrix<T, 3, 1> refMagVector)
	{
		Eigen::Matrix<T, 4, 1> result;
		Eigen::Matrix<T, 4, 3> leftProduct;
		Eigen::Matrix<T, 3, 1> rightProduct;

		T qw = estMagQuaternion.w();
		T qx = estMagQuaternion.x();
		T qy = estMagQuaternion.y();
		T qz = estMagQuaternion.z();
		T vmx = measuredMagVector[0];
		T vmy = measuredMagVector[1];
		T vmz = measuredMagVector[2];
		T vrx = refMagVector[0];
		T vry = 0;
		T vrz = refMagVector[2];
		
		leftProduct << 2 * (
			( vrx*qw - vrz*qy), (-vrx*qz + vrz*qx), (vrx*qy + vrz*qw),
			( vrx*qx + vrz*qz), ( vrx*qy + vrz*qw), (vrx*qz - vrz*qx),
			(-vrx*qy - vrz*qw), ( vrx*qx + vrz*qz), (vrx*qw - vrz*qy),
			(-vrx*qz + vrz*qx), (-vrx*qw + vrz*qy), (vrx*qx + vrz*qz)
			);

		rightProduct <<
			vrx * (qw*qw + qx*qx - qy*qy - qz*qz) + vrz * (-2*qw*qy + 2*qx*qz)            - vmx,
			vrx * (-2*qw*qz + 2*qx*qy)			  + vrz * (2*qw*qx + 2*qy*qz)			  - vmy,
			vrx * (2*qw*qy + 2*qx*qz)			  + vrz * (qw*qw - qx*qx - qy*qy + qz*qz) - vmz;

		result = leftProduct * rightProduct;
		return Eigen::Quaternion<T>(result[0], result[1], result[2], result[3]);
	}

};

#endif /* !ENHANCED_MADGWICK_HPP */