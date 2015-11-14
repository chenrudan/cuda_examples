#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <set>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include "decision_tree.hpp"
#include "tools.hpp"


Pars mypar;

double getGini(thrust::device_vector<double> &number_in_each_class){

	square<double> unary_op;
	thrust::plus<double> binary_op;

	double sum = thrust::reduce(number_in_each_class.begin(), \
			number_in_each_class.end(), (double)0, thrust::plus<double>());
	thrust::device_vector<double> vec_sum(number_in_each_class.size(), (double)sum);
	thrust::device_vector<double> vec_divides(number_in_each_class.size(), (double)0);

	thrust::transform(number_in_each_class.begin(), \
			number_in_each_class.end(), vec_sum.begin(), vec_divides.begin(), \
			thrust::divides<double>());

	sum = thrust::transform_reduce(vec_divides.begin(), vec_divides.end(), \
			unary_op, (double)0, binary_op);

	//	thrust::copy(vec_divides.begin(), vec_divides.end(), std::ostream_iterator<double>(std::cout, "\n"));
	return 1-sum;
}

///number_in_branch里面保存的就是每个类的个数
///
///
double getGiniOfFeature(thrust::device_vector<double> &number_in_branch_1, \
		thrust::device_vector<double> &number_in_branch_2 ){
	const int all_number = number_in_branch_1.size() + number_in_branch_2.size();
	std::cout << ((double)number_in_branch_1.size()/all_number)*getGini(number_in_branch_1) \
		<< ":" << ((double)number_in_branch_2.size()/all_number)*getGini(number_in_branch_2) << std::endl;
	return ((double)number_in_branch_1.size()/all_number)*getGini(number_in_branch_1) \
		+ ((double)number_in_branch_2.size()/all_number)*getGini(number_in_branch_2);
}

///通过某个特征的取值，来将输入set集分成两个输出set集，和两个累积各个类个数的vector
///由于是cart，它只判断是否
///输入的branch和number_in_branch都要初始化为0
///
void splitToBranchForOneFeature(int feature_pos, double feature_value, \
		std::set< FeatureValueAndClass<double> > &in, \
		std::set< FeatureValueAndClass<double> > &branch1, \
		std::set< FeatureValueAndClass<double> > &branch2, \
		thrust::device_vector<double> &number_in_branch_1, \
		thrust::device_vector<double> &number_in_branch_2){
	for(std::set< FeatureValueAndClass<double> >::iterator it = in.begin(); \
			it!=in.end(); ++it){
		//		std::cout << (*it);
		if((*it).getOneFeatureValue(feature_pos) == feature_value){
			branch1.insert(*it);
			number_in_branch_1[(*it).getClass()]++;
		}else{
			branch2.insert(*it);
			number_in_branch_2[(*it).getClass()]++;
		}
	}
}

void subBuildDecisionTree(std::set< FeatureValueAndClass<double> > &cur_all_sample, \
		DecisionTree<double> &dt, thrust::host_vector<int> poses, \
		thrust::host_vector<int> pos_path, thrust::host_vector<double> value_path, \
		double last_min_gini){


	if(poses.size() == 0 || last_min_gini == 0)
		return;

	thrust::host_vector<double> gini_of_pos; ///>选择每一个剩下的特征进行切分得到的gini值
	std::set< FeatureValueAndClass<double> > branch1;
	std::set< FeatureValueAndClass<double> > branch2;
	thrust::device_vector<double> number_in_branch_1(mypar.num_class, 0);
	thrust::device_vector<double> number_in_branch_2(mypar.num_class, 0);

	for(int i=0; i<poses.size(); i++){
		splitToBranchForOneFeature(poses[i], 1, cur_all_sample, branch1, branch2, \
				number_in_branch_1, number_in_branch_2);

		thrust::device_vector<double>::iterator it = remove(number_in_branch_1.begin(), \
				number_in_branch_1.end(), 0);
		number_in_branch_1.resize(it - number_in_branch_1.begin());
		it = remove(number_in_branch_2.begin(), number_in_branch_2.end(), 0);
		number_in_branch_2.resize(it - number_in_branch_2.begin());

		gini_of_pos.push_back(getGiniOfFeature(number_in_branch_1, \
					number_in_branch_2));
		number_in_branch_1.assign(mypar.num_class, 0);
		number_in_branch_2.assign(mypar.num_class, 0);
		branch1.clear();
		branch2.clear();
	}	
	///返回gini值中比阈值大的位置，大就直接退出
	thrust::host_vector<double>::iterator it = thrust::upper_bound( \
			gini_of_pos.begin(), gini_of_pos.end(), mypar.gini_threshold);
	if(it != gini_of_pos.end())
		return;

	it = thrust::min_element(gini_of_pos.begin(), gini_of_pos.end());
	const int least_gini_pos = it - gini_of_pos.begin();
	pos_path.push_back(poses[least_gini_pos]);

	splitToBranchForOneFeature(poses[least_gini_pos], 1, cur_all_sample, branch1, branch2, \
			number_in_branch_1, number_in_branch_2);
	

	poses.erase(poses.begin()+least_gini_pos);

	/*
		std::cout << "number in branch1:\t";
		thrust::copy(number_in_branch_1.begin(), number_in_branch_1.end(), \
				std::ostream_iterator<double>(std::cout, "\t"));
		std::cout << "\n";
		std::cout << "number in branch2:\t";
		thrust::copy(number_in_branch_2.begin(), number_in_branch_2.end(), \
				std::ostream_iterator<double>(std::cout, "\t"));
		std::cout << "\n";
	*/
	std::cout << branch1 << branch2;

		std::cout << "pos path:\t";
		thrust::copy(pos_path.begin(), pos_path.end(), \
				std::ostream_iterator<int>(std::cout, "\t"));
		std::cout << "\n";

	value_path.push_back(1);

		std::cout << "value path:\t";
		thrust::copy(value_path.begin(), value_path.end(), \
				std::ostream_iterator<double>(std::cout, "\t"));
		std::cout << "\n";

	std::cout << "smallest gini:" << *it << "\n";
	
	thrust::device_vector<double>::iterator number_it = thrust::max_element( \
			number_in_branch_1.begin(), number_in_branch_1.end());
	
	dt.insert(pos_path, value_path, number_it-number_in_branch_1.begin());
	subBuildDecisionTree(branch1, dt, poses, pos_path, value_path, *it);
	
	number_it = thrust::max_element(number_in_branch_2.begin(), number_in_branch_2.end());

	value_path.pop_back();
	value_path.push_back(0);
	dt.insert(pos_path, value_path, number_it-number_in_branch_1.begin());
	subBuildDecisionTree(branch2, dt, poses, pos_path, value_path, *it);
}

///递归的来根据不同的特征取值来划分，暂时特征的取值为0，1
///
///
void buildDecisionTree(std::set< FeatureValueAndClass<double> > &all_sample){

	DecisionTree<double> dt;
	thrust::host_vector<int> poses; ///>用来保存每一轮剩下能够选择的feature位置，值1为左分支，0为右分支
	thrust::host_vector<double> value_path; 
	thrust::host_vector<int> pos_path; ///>保存的是已经选择的feature和它对应的值

	for(int i=0; i<(*all_sample.begin()).getNumFeatures(); i++){
		poses.push_back(i);
	}

	subBuildDecisionTree(all_sample, dt, poses, pos_path, value_path, 1);
	dt.printTree();

}

int main(int argc, char** argv){

	std::cout << "argc: " << argc << std::endl;
	std::cout << "argv[0]: " << argv[0] << std::endl;

	int num_features;
	std::stringstream ss;
	ss << argv[1];
	ss >> num_features;
	ss.clear();
	ss << argv[2];
	ss >> mypar.num_class;
	ss.clear();
	ss << argv[3];
	ss >> mypar.gini_threshold;
	ss.clear();

	std::set< FeatureValueAndClass<double> > all_sample;
	std::ifstream fin_data(argv[4]);
	std::ifstream fin_label(argv[5]);
	int k = 0;
	thrust::device_vector<double> features(num_features);
	char output[100];
	int label;
	double tmp;
	std::cout << "f1\tf2\tf3\tlabel\n";
	while(fin_data>>output){
		ss << output;
		ss >> tmp;
		features[k%num_features] = tmp;
		std::cout << features[k%num_features] << "\t";
		ss.clear();
		k++;
		if(k % num_features == 0 && k != 0){
			fin_label >> output;
			ss << output;
			ss >> label;
			std::cout << label;
			ss.clear();
			FeatureValueAndClass<double> s(features, k/num_features, label);
			std::cout << "\n";
			all_sample.insert(s);
		}
	}
	std::cout << *all_sample.begin();
	std::cout << all_sample;
	buildDecisionTree(all_sample);	

	return 0;
}


























