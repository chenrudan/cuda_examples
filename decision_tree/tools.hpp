
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <set>


template <typename T> 
struct square { 
	__host__ __device__ T operator()(const T& x) const 
	{ 
		return x * x;  
	}  
};

typedef struct PARAMETER{
	int num_class;
	double gini_threshold;
	int sample_threshold;
}Pars;

template<typename Dtype>
class FeatureValueAndClass;

template<typename Dtype>
std::ostream& operator<<(std::ostream &output, const FeatureValueAndClass<Dtype> &f);

///这个保存的是每条数据的每个特征的取值和对应的class
///
///
template<typename Dtype>
class FeatureValueAndClass{
private:
	int _id;
	thrust::device_vector<Dtype> _value;
	int _class_id;
public:

	FeatureValueAndClass(thrust::device_vector<Dtype> &host, \
			const int id, const int class_id) : _id(id), _class_id(class_id){
		_value.resize(host.size());
		thrust::copy(host.begin(), host.end(), _value.begin());
	}

	Dtype getOneFeatureValue(const int pos) const{
		return _value[pos];
	}
	int getClass() const{
		return _class_id;
	}
	int getNumFeatures() const{
		return _value.size();
	}
	thrust::device_vector<Dtype>& getAllFeatureValue() {
		return _value;
	}
	bool operator<(const FeatureValueAndClass<Dtype> &other) const{
		return _id < other._id;
	}
	int getId() const{
		return _id;
	}
	friend std::ostream& operator<< <>(std::ostream &output, const FeatureValueAndClass<Dtype> &f);
	
};

template<typename Dtype>
std::ostream& operator<<(std::ostream &output, \
		const FeatureValueAndClass<Dtype> &f){
	for(int i=0; i<f.getNumFeatures(); i++){
		output << f.getOneFeatureValue(i) << "\t";
	}
	output << std::endl;
	return output;
}

std::ostream& operator<<(std::ostream &output, \
		const std::set< FeatureValueAndClass<double> > &s){
	output << "set element: ";
	for(std::set< FeatureValueAndClass<double> >::iterator it = s.begin(); \
			it!=s.end(); ++it){
		output << (*it).getId() << "\t";
	}
	output << std::endl;
	return output;

}
