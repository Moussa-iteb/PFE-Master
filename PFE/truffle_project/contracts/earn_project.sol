// SPDX-License-Identifier: MIT

pragma solidity ^0.8.9;

contract earn_project {
    struct Data {
        string zone;
        string ID_noeud;
        uint256 Date;
        bytes32 dataHash; // Ajout d'un champ pour stocker le hachage
    }

    mapping(uint256 => Data) private dataByDate;
    uint256[] private allDates;

    function storeData(string memory _zone, string memory _ID_noeud, uint256 _Date) public {
        require(dataByDate[_Date].Date == 0, "Data already exists for this timestamp");

        bytes32 dataHash = keccak256(abi.encodePacked(_zone, _ID_noeud, _Date)); // Calcul du hachage
        Data storage newData = dataByDate[_Date];
        newData.zone = _zone;
        newData.ID_noeud = _ID_noeud;
        newData.Date = _Date;
        newData.dataHash = dataHash; // Stockage du hachage

        allDates.push(_Date);
    }

    function getDataByDate(uint256 _timestamp) public view returns (string memory, string memory, uint256, bytes32) {
        require(dataByDate[_timestamp].Date != 0, "No data found for this timestamp");

        Data memory existingData = dataByDate[_timestamp];
        return (existingData.zone, existingData.ID_noeud, existingData.Date, existingData.dataHash);
    }

    function getAllTransactions() public view returns (Data[] memory) {
        Data[] memory transactions = new Data[](allDates.length);
        
        for (uint256 i = 0; i < allDates.length; i++) {
            transactions[i] = dataByDate[allDates[i]];
        }

        return transactions;
    }
}
