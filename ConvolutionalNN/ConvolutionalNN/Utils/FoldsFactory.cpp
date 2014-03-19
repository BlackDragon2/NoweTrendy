
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
	acquireFitTactic(pFoldsCount, pFitTactic, *v);

	size_t to = std::min(pElementsCount, v->size());
	for(size_t i=0UL; i<to; ++i)
		(*v)[i] = i;

	for(size_t i=pElementsCount; i<v->size(); ++i)
		(*v)[i] = rand() % pElementsCount;

	std::random_shuffle(v->begin(), v->end());
	return v;
}


std::shared_ptr<std::vector<size_t>> FoldsFactory::prepareFoldVectorWithCopies(
	size_t		pElementsCount, 
	size_t		pFoldsCount, 
	FitTactic	pFitTactic)
{
	std::shared_ptr<std::vector<size_t>> v(new std::vector<size_t>(pElementsCount));
	acquireFitTactic(pFoldsCount, pFitTactic, *v);

	for(size_t i=0UL; i<v->size(); ++i)
		(*v)[i] = rand() % v->size();
	
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
	
	if(pFitTactic == FitTactic::CUT)
		pVector.resize(pVector.size() - rest);
	else if(pFitTactic == FitTactic::EXTEND_WITH_COPIES)
		pVector.resize(pVector.size() + pFoldsCount - rest);
}



	}
}