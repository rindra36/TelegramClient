async function searchPocketOptionUID(uid) {
    try {
        const response = await fetch("https://pocketoption.com/en/api/social-modal/search-user ", {
            method: "POST",
            credentials: "include",
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "en-US,en;q=0.5",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Isdemo": "0",
                "Istournament": "0",
                "Ischart": "1",
                "X-Requested-With": "XMLHttpRequest",
                "Sec-GPC": "1",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin"
            },
            referrer: "https://pocketoption.com/en/cabinet/ ",
            body: `q=${encodeURIComponent(uid)}`
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error fetching data:", error);
        return null;
    }
}

function generateOldCombination() {
  let digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  let combinations = [];
  
  digits.forEach(d1 => {
    digits.forEach(d2 => {
      digits.forEach(d3 => {
        combinations.push(`97${d1}${d2}7${d3}12`)
      });
    });
  });
  
  return combinations;
}

async function generateCombination() {
    let digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    let digits1 = ['3', '5', '9'];
    let digits2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
    let digits3 = ['1', '4', '7'];
  	let digits4 = ['0', '3', '5', '8', '9']
    let digits5 = ['1', '4']
    let combinations = [];
  
  	const oldCombinations = generateOldCombination()
    
    digits1.forEach(d1 => {
      digits2.forEach(d2 => {
        digits3.forEach(d3 => {
          digits4.forEach(d4 => {
            digits5.forEach(d5 => {
              let combination = `97${d1}${d2}${d3}${d4}${d5}2`
              if (oldCombinations.includes(combination) === false) {
                combinations.push(combination)
              }
            });
          });
        });
      });
    });
  
  	for (const uid of combinations) {
      await searchPocketOptionUID(uid)
    }
}

await generateCombination()