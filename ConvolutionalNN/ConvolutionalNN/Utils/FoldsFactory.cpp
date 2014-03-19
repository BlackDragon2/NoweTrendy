
#include <algorithm>

#include "FoldsFactory.h"


namespace cnn {
	namespace utils {


std::shared_ptr<std::vector<size_t>> FoldsFactory::prepareFoldVector(
	size_t		pElementsCount, 
	size_t		pFoldsCount, 
	FitTactic	pFitTactic)
{
	std::shared_ptr<std::vector<size_t>> v(new std::vector<size_t>(pElementsCount));
	for(size_t i=0UL; i<v->size(); ++i)
		(*v)[i] = i;
	acquireFitTactic(pFoldsCount, pFitTactic, *v);
	std::random_shuffle(v->begin(), v->end());
	return v;
}

	
void FoldsFactory::acquireFitTactic(
	size_t					pFoldsCount, 
	FitTactic				pFitTactic,
	std::vector<size_t>&	pVector)
{
	size_t rest = pVector.size() % pFoldsCount;
	if(rest == 0UL || pFitTactic == FitTactic::DEFAULT)
		return;

	if(pFitTactic == FitTactic::CUT){
		pVector.resize(pVector.size() - rest);
		return;
	}

	size_t oldSize = pVector.size();
	size_t newSize = pVector.size() + pFoldsCount - rest;
	pVector.resize(newSize);

	if(pFitTactic == FitTactic::EXTEND)
		for(size_t i=oldSize; i<newSize; ++i)
			pVector[i] = oldSize++;
	else if(pFitTactic == FitTactic::EXTEND_WITH_COPIES)
		for(size_t i=oldSize; i<newSize; ++i)
			pVector[i] = rand() % oldSize;
}


	}
}