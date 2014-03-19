
#include <memory>
#include <vector>


namespace cnn {
	namespace utils {


class FoldsFactory {
public:
	enum FitTactic {
		DEFAULT,
		CUT,
		EXTEND_WITH_COPIES
	};


public:
	static std::shared_ptr<std::vector<size_t>> prepareFoldVector(
		size_t		pElementsCount, 
		size_t		pFoldsCount, 
		FitTactic	pFitTactic = FitTactic::DEFAULT);

	static std::shared_ptr<std::vector<size_t>> prepareFoldVectorWithCopies(
		size_t		pElementsCount, 
		size_t		pFoldsCount, 
		FitTactic	pFitTactic);

	
private:
	static void acquireFitTactic(
		size_t					pFoldsCount, 
		FitTactic				pFitTactic,
		std::vector<size_t>&	pVector);
};


	}
}