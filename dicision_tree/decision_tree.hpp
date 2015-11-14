
#include <thrust/host_vector.h>
#define HAVE_NO_CLASS 200
///
///这个数据结构保存的是树的每一个节点，存放这个节点的id，对应的feature和取值
template<typename Dtype>
struct TreeNode{
	int _cur_feature_pos;
	Dtype _cur_feature_value;
	int _cur_class;
	TreeNode<Dtype> *_l_child;
	TreeNode<Dtype> *_r_child;
};

///
///树的数据结构
template<typename Dtype>
class DecisionTree{
	public:
		DecisionTree(){
			_root = NULL;
		}
		~DecisionTree();

		void insert(thrust::host_vector<int> features, \
				thrust::host_vector<Dtype> values, \
				const int class_id);
		void destoryTree(TreeNode<Dtype> *node);
		void printTree();
	private:
		TreeNode<Dtype> *_root;
		void printNode(TreeNode<Dtype> *node);
};

template<typename Dtype>
DecisionTree<Dtype>::~DecisionTree(){
	destoryTree(_root);
}

template<typename Dtype>
void DecisionTree<Dtype>::destoryTree(TreeNode<Dtype> *node){
	if(node != NULL){
		destoryTree(node->_l_child);
		destoryTree(node->_r_child);
		delete node;
	}
}

template<typename Dtype>
void DecisionTree<Dtype>::printNode(TreeNode<Dtype> *node){
	std::cout << "\t";
	std::cout << node->_cur_feature_pos;
	if(node->_cur_class != HAVE_NO_CLASS)
		std::cout << "(" << node->_cur_class << ")\n";
	std::cout << "\n";
	if(node->_l_child){
		printNode(node->_l_child);
	}
	std::cout << "\t";
	if(node->_r_child){
		printNode(node->_r_child);
	}
}

template<typename Dtype>
void DecisionTree<Dtype>::printTree(){
	printNode(_root);	
	std::cout << "\n";
}

template<typename Dtype>
void DecisionTree<Dtype>::insert(thrust::host_vector<int> poses, \
		thrust::host_vector<Dtype> values, const int class_id){
	if(poses.size() != 0){
		TreeNode<Dtype> *current_node = _root;
		if(current_node == NULL){
			_root = new TreeNode<Dtype>;
			_root->_cur_feature_pos = poses[0];
			_root->_cur_feature_value = values[0];
			_root->_cur_class = class_id;
			_root->_l_child = NULL;
			_root->_r_child = NULL;
		}else{
			for(int i=0; i<poses.size(); i++){
				if(poses[i] == current_node->_cur_feature_pos){
					if(values[i] == current_node->_cur_feature_value){
						if(current_node->_l_child != NULL){
							current_node->_cur_class = HAVE_NO_CLASS;
							current_node = current_node->_l_child;
						}else{
							current_node->_l_child = new TreeNode<Dtype>;
							current_node->_l_child->_cur_feature_pos = poses[i];
							current_node->_l_child->_cur_feature_value = values[i];
							current_node->_cur_class = class_id;
							current_node->_l_child->_l_child = NULL;
							current_node->_l_child->_r_child = NULL;
							return;
						}
					}else{
						if(current_node->_r_child != NULL){
							current_node->_cur_class = HAVE_NO_CLASS;
							current_node = current_node->_r_child;
						}else{
							current_node->_r_child = new TreeNode<Dtype>;
							current_node->_r_child->_cur_feature_pos = poses[i];
							current_node->_r_child->_cur_feature_value = values[i];
							current_node->_r_child->_l_child = NULL;
							current_node->_r_child->_r_child = NULL;
							return;
						}
					}
				}else{
					std::cout << "feature position is not correct!\n";
				}
			}
		}
	}
}

