# Deanonymising Households Trading on a BlockchainSmart Grid
Engineering Computer and Software Systems Honours Thesis
Andrew Mather

**Abstract**
This project aims to contribute to the study of user anonymity in blockchain for the Internet of Things. The research explores machine learning methods to deanonymise users on a smart grid with blockchain. Data stored on decentralised blockchains is permanent and user privacy can be at risk. Research exists into anonymity concerns in blockchain but IoT and smart grid contexts warrant further research.

The project analysis includes stored blockchain transactions and off-chain solar exposure data. Section one and two will investigate the research topic and the surrounding literature. Followed by the third covering the project's methodology. It aims to determine how effective machine learning is to deanonymise smart grid blockchain users. This will highlight long-term blockchain anonymity risks due to permanent transaction data. Section four will report these results against the objectives.

The approach will first, source energy grid data and construct blockchain ledgers. Second, apply machine learning analysis on past blockchain transactions to identify households. Third, measure the benefit of adding off-chain solar exposure data as households also produce solar energy. Last, the project will investigate public key and ledger obfuscation methods to improve user privacy. Project outcomes include a better understanding of user privacy risks but also methods to mitigate these risks.

Results found machine learning models accurately link user transactions from past data without privacy steps. More frequent transactions aid attackers and the peak classification accuracy was 84% (customer ID) and 63% (postcode) with a convolutional network. Attacker success rates improve by including solar exposure data. Household energy usage and solar generation were reconstructed from net export at 85% R^2 accuracy with regression models. Smarter users can significantly reduce attack success by adding public keys. The lowest result became 4% (customer ID) and 11% (postcode). There are clear benefits in public keys up to about 20, where the benefit sharply tapers off thereafter and may not be worth the cost for a user. There is less notable but still a privacy benefit as public keys per ledger increase.

The results highlight concern for a user's privacy without appropriate obfuscation methods. If a grid participant uses a single public key, there are serious issues with maintaining privacy between transactions and their anonymity. While blockchain guarantees some level of anonymity, it is not absolute. Attackers can find creative ways of using stored blockchain data without even accessing a real-time network. A standard blockchain poses risks to users in this research's context and warrants more attention.

**Code**

Update dependencies: pipreqs /path/to/project --force

Run scripts with Terminal or Powershell, avoid an IDE as too slow.

Runnable scripts include --help or will print a message upon incorrect usage for guidance.


pop\_blockchain.py is used to prepare datasets.

solar.py is used to prepare solar exposure data including datasets.

obfuscation.py is used to create obfuscated datasets from standard ones.

run\_analysis.py is used to run analysis with many selectable options detailed in its help.

preprocess.py and ml\_methods are used by run analysis when running. preprocess does what you think, ml\_methods includes machine learning models implementations.

graphs.py was used to create graphs of the results.
