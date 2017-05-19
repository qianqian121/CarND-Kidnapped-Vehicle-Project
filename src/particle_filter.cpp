/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // This line creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 320;
  weights.resize(num_particles);

  default_random_engine gen;
  for (int i = 0; i<num_particles; i++) {

    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (vector<Particle>::iterator it=particles.begin(); it!=particles.end(); ++it) {
    Particle &p = *it;

    if (fabs(yaw_rate) < 0.01) {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
      p.theta += yaw_rate * delta_t;
    } else {
      p.x += (velocity/yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      p.theta += yaw_rate * delta_t;
    }

    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  std::vector<LandmarkObs> ret;

  for (int i=0; i<observations.size(); i++) {
    double minimal_distance = INFINITY;
    int thej = -1;

    for (int j=0; j<predicted.size(); j++) {

      double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      // find the closed measurement for each observed measurement
      if (distance < minimal_distance) {
        minimal_distance = distance;
        thej = j;
      }
    }

    // thej must be valid
    ret.push_back(predicted[thej]);
  }

  return ret;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  for (vector<Particle>::iterator it=particles.begin(); it!=particles.end(); ++it) {
  Particle &p = *it;

    // find predicted landmarks associated with current particle
    std::vector<LandmarkObs> predicted;
    for (int i=0; i<map_landmarks.landmark_list.size(); i++) {
      double x = map_landmarks.landmark_list[i].x_f;
      double y = map_landmarks.landmark_list[i].y_f;
      if (dist(x,y, p.x, p.y) <= sensor_range) {
        LandmarkObs o;
        o.id = map_landmarks.landmark_list[i].id_i; 
        o.x = x;
        o.y = y;
        predicted.push_back(o);
      }
    }

    // transform observations from local to global
    vector<LandmarkObs> transformed;
    for (int i=0; i<observations.size(); i++) {
        LandmarkObs o = observations[i];
        double x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
        double y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
        o.x = x; o.y = y;
        transformed.push_back(o);
    }

    // find associated for observations
    std::vector<LandmarkObs> associated = dataAssociation(predicted, transformed);

    // calculate probability based on mult-variate gaussian distribution
    double P = 1.0;
    for (int i=0; i<transformed.size(); i++) {
        double x0 = transformed[i].x;
        double y0 = transformed[i].y;
        double x1 = associated[i].x;
        double y1 = associated[i].y;
        double cx = std_landmark[0];
        double cy = std_landmark[1];
        double likelyhood = (x0-x1)*(x0-x1)/(cx*cx) + (y0-y1)*(y0-y1)/(cy*cy);
        likelyhood = exp(-likelyhood/2) / (2 * M_PI * cx * cy);
        P *= likelyhood;
    }

    // update the weight for the particle
    p.weight = P;
    weights[p.id] = P;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  std::vector<Particle> particle_resampled;
  std::discrete_distribution<int> rdist(weights.begin(), weights.end());

  for (int i=0; i< num_particles; i++) {
    // get one item from distribution
    int w = rdist(gen);
    particle_resampled.push_back(particles[w]);
    //reset id
    particle_resampled[i].id=i;
  }
  
  particles = std::move(particle_resampled);
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
