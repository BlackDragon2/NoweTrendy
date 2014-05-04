
#ifndef CNN_FOLDS_FACTORY_H_
#define CNN_FOLDS_FACTORY_H_


#include <memory>
#include <vector>


namespace cnn {
	namespace utils {


class FoldsFactory {
public:
	typedef std::vector<size_t>			Sequence;
	typedef std::shared_ptr<Sequence>	SequencePtrS;
	typedef std::vector<Sequence>		Folds;
	typedef std::shared_ptr<Folds>		FoldsPtrS;


	enum FitTactic {
		DEFAULT,
		CUT,
		EXTEND_WITH_COPIES
	};


public:
	static SequencePtrS prepareSequence(
		size_t		pElementsCount, 
		size_t		pFoldsCount, 
		FitTactic	pFitTactic = FitTactic::DEFAULT);

	static SequencePtrS prepareSequenceWithCopies(
		size_t		pElementsCount, 
		size_t		pFoldsCount, 
		FitTactic	pFitTactic = FitTactic::DEFAULT);

	static FoldsPtrS prepareFolds(
		SequencePtrS const& pSequence,
		size_t				pFoldsCount);

	
private:
	static size_t acquireFitTactic(
		size_t		pElementsCount,
		size_t		pFoldsCount, 
		FitTactic	pFitTactic);
};


	}
}


#endif	/* #ifndef CNN_FOLDS_FACTORY_H_ */