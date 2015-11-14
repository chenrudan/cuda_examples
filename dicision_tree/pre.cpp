
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>

static inline double computeSquare(double x){ 
	return x*x;
}

///这个保存的是每条数据的每个特征的取值和对应的class
///
///
template<typename Dtype>
class FeatureValueAndClass{
private:
	std::vector<Dtype> _value;
	int _class_id;
public:
	Dtype getOneFeatureValue(int pos){
		return _value[pos];
	}
	int getClass(){
		return _class_id;
	}
};

double getGini(std::vector<double> &number_in_each_class){
	double sum = std::accumulate(number_in_each_class.begin(), \
			number_in_each_class.end(), 0.0);
	std::vector<double> vec_sum(number_in_each_class.size(), sum);
	std::vector<double> vec_divides(number_in_each_class.size(), 0);

	std::transform(number_in_each_class.begin(), \
			number_in_each_class.end(), vec_sum.begin(), vec_divides.begin(), \
			std::divides<double>());
	
	std::transform(vec_divides.begin(), vec_divides.end(), vec_sum.begin(), \
			computeSquare);

	
	sum = std::accumulate(vec_sum.begin(), vec_sum.end(), 0.0);
/*
	std::cout << "ori data:\n";
	for(int i = 0; i < number_in_each_class.size(); i++){
		std::cout << number_in_each_class[i] << std::endl;
	}
	std::cout << "sum:\n";
	for(int i = 0; i < vec_sum.size(); i++){
		std::cout << vec_sum[i] << std::endl;
	}
	std::cout << "divides:\n";
	for(int i = 0; i < vec_divides.size(); i++){
		std::cout << vec_divides[i] << std::endl;
	}
	std::cout << sum << std::endl;
*/
	return sum;
}

///number_in_branch里面保存的就是每个类的个数
///
///
double getGiniOfFeature(std::vector<double> &number_in_branch_a, \
		std::vector<double> &number_in_branch_b ){
	const int all_number = number_in_branch_a.size() + number_in_branch_b.size();
	return ((double)number_in_branch_a.size()/all_number)*getGini(number_in_branch_a) \
		+ ((double)number_in_branch_b.size()/all_number)*getGini(number_in_branch_b);
}

int main(int argc, char** argv){

	std::cout << "argc: " << argc << std::endl;
	std::cout << "argv[0]: " << argv[0] << std::endl;

	std::vector<double> number_in_each_class(10);
	for(int i = 0; i < number_in_each_class.size(); i++){
		number_in_each_class[i] = i;
	}

	double all_fini = getGini(number_in_each_class);
	std::cout << all_fini << std::endl;

	std::vector<double> number_in_branch_a(10);
	std::vector<double> number_in_branch_b(10);
	for(int i = 0; i < number_in_branch_a.size(); i++){
		number_in_branch_a[i] = i;
	}
	for(int i = 0; i < number_in_branch_b.size(); i++){
		number_in_branch_b[i] = i;
	}
	all_fini = getGiniOfFeature(number_in_branch_a, number_in_branch_b);
	std::cout << all_fini << std::endl;

	return 0;
}
