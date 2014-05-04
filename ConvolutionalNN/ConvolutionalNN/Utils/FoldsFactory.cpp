
#include <algorithm>

#include "FoldsFactory.h"
#include "Utils.h"


namespace cnn {
	namespace utils {


FoldsFactory::SequencePtrS FoldsFactory::prepareSequence(
	size_t		pElementsCount, 
	size_t		pFoldsCount, 
	FitTactic	pFitTactic)
{
	std::shared_ptr<std::vector<size_t>> v(new std::vector<size_t>(
		acquireFitTactic(pElementsCount, pFoldsCount, pFitTactic)));

	size_t to = std::min(pElementsCount, v->size());
	for(size_t i=0UL; i<to; ++i)
		(*v)[i] = i;

	for(size_t i=to; i<v->size(); ++i)
		(*v)[i] = bigRand64() % pElementsCount;

	std::random_shuffle(v->begin(), v->end());
	return v;
}


FoldsFactory::SequencePtrS FoldsFactory::prepareSequenceWithCopies(
	size_t		pElementsCount, 
	size_t		pFoldsCount, 
	FitTactic	pFitTactic)
{
	std::shared_ptr<std::vector<size_t>> v(new std::vector<size_t>(
		acquireFitTactic(pElementsCount, pFoldsCount, pFitTactic)));

	for(size_t i=0UL; i<v->size(); ++i)
		(*v)[i] = bigRand64() % pElementsCount;
	
	return v;
}


FoldsFactory::FoldsPtrS FoldsFactory::prepareFolds(
	SequencePtrS const& pSequence,
	size_t				pFoldsCount)
{
	FoldsPtrS folds(new Folds(pFoldsCount, Sequence()));
	for(size_t f=0UL; f<pFoldsCount; ++f)
		(*folds)[f].reserve(static_cast<size_t>(std::ceil(
			static_cast<double>(pSequence->size()) / pFoldsCount)));

	for(size_t s=0UL; s<pSequence->size(); ++s)
		(*folds)[s % pFoldsCount].push_back((*pSequence)[s]);

	return folds;
}

	
size_t FoldsFactory::acquireFitTactic(
	size_t		pElementsCount,
	size_t		pFoldsCount, 
	FitTactic	pFitTactic)
{
	size_t rest = pElementsCount % pFoldsCount;	
	if(pFitTactic == FitTactic::CUT)
		return pElementsCount - rest;
	else if(pFitTactic == FitTactic::EXTEND_WITH_COPIES)
		return pElementsCount + pFoldsCount - rest;
	return pElementsCount;
}



	}
}