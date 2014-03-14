
#pragma once

#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>


namespace utils {


template<typename T>
void createFolds(
	std::vector<T> const&			pInput, 
	std::vector<std::vector<T>>&	pOutput,
	size_t							pFoldsCount)
{
	pOutput.clear();
	pOutput.resize(pFoldsCount);

	std::vector<size_t> indices(pInput.size());
	for(size_t i=0UL; i<pInput.size(); ++i)
		indices[i] = i;

	std::random_shuffle(indices.begin(), indices.end());
}


template <typename T>
size_t createFoldsInPlace(std::vector<T>& pData, size_t pFoldsCount){
	std::random_shuffle(pData.begin(), pData.end());
	return std::ceil(static_cast<double>(pData.size()) / pFoldsCount);
}


}