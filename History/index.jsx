import React, { useEffect, useState } from "react";
import { DataGrid } from "@mui/x-data-grid";
import Web3 from "web3";
import YourContract from "./earn_project.json";

const History = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchAllTransactions = async () => {
      try {
        const web3 = new Web3("http://localhost:9545");
        const networkId = await web3.eth.net.getId();
        const deployedNetwork = YourContract.networks[networkId];
        const contract = new web3.eth.Contract(YourContract.abi, deployedNetwork && deployedNetwork.address);
  
        const result = await contract.methods.getAllTransactions().call();
        console.log("Données récupérées :", result);
  
        if (result && result.length > 0) {
          const transactions = result.map((data, index) => {
            return {
              id: index + 1,
              Date: new Date(parseInt(data[2]) * 1000).toLocaleDateString(),
              ID_noeud: data[1].toString(), // Convertir en chaîne de caractères
              zone: data[0],
            };
          });
          setData(transactions);
        } else {
          console.error("Aucune donnée trouvée.");
        }
      } catch (error) {
        console.error("Erreur lors de la récupération des données :", error);
      }
    };
  
    fetchAllTransactions();
  }, []);
  
  const columns = [
    { field: "Date", headerName: "Date", width: 200 },
    { field: "ID_noeud", headerName: "ID_noeud", width: 200 },
    { field: "zone", headerName: "Zone", width: 200 },
  ];

  return (
    <div style={{ height: 400, width: '100%' }}>
      <h1>Historique</h1>
      <DataGrid rows={data} columns={columns} autoHeight />
    </div>
  );
};

export default History;
