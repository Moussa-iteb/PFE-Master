const Web3 = require('web3');
const earn_project = artifacts.require("earn_project");


contract("earn_project", accounts => {
  let contractInstance;

  before(async () => {
    contractInstance = await earn_project.deployed();
  });

  it("should change the date in the blockchain", async function () {
    const result = await contractInstance.storeData("zone1", "noued1", 123, { from: accounts[0] });
    assert.equal(result.receipt.status, true, "La transaction a échoué");
  });

  it("should retrieve the date from the blockchain", async function () {
    const result = await contractInstance.getDataByDate(123);
    const Date = result[2].toNumber();
    assert.equal(Date, 123, "Le timestamp récupéré ne correspond pas à la valeur attendue");
    const zone = result[0];
    const ID_noeud = result[1];

    console.log(zone, ID_noeud, Date); 
  });
});
