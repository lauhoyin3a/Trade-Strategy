import { useEffect, useState } from "react";
import boat from './assets/boat.png';
import p1 from './assets/p_image/p1.png';
import p2 from './assets/p_image/p2.png';
import p3 from './assets/p_image/p3.png';
import p4 from './assets/p_image/p4.png';
import e1 from './assets/e_image/e1.png';
import e2 from './assets/e_image/e2.png';
import './Game.css';


const Game = () => {
    const [gameStarted, setGameStarted] = useState(true);
    const [gameEnded, setGameEnded] = useState(false);
    const [score, setScore] = useState(0);
    const [timer, setTimer] = useState(6);
    const [catcherPosition, setCatcherPosition] = useState(0);
    const [dropPosition, setDropPosition] = useState(0);
    const [dropImage, setDropImage] = useState(p1);
    const [userName, setUserName] = useState("");
    const screenHeight = window.innerHeight;
    const screenWidth = window.innerWidth;
    const dropUnit = screenHeight * 0.01;

    const pImageArray = [p1, p2, p3, p4];
    const eImageArray = [e1, e2]

    useEffect(() => {
      let interval = null;
     
      if (gameStarted && !gameEnded && timer > 0) {
        interval = setInterval(() => {
          setTimer((prevTimer) => prevTimer - 0.1);
        }, 100);
        
        if (dropPosition + dropUnit < screenHeight){
          
          dropAction();
        }
        
      }
  
      if (Math.ceil(timer) === 0) {
        setGameEnded(true);
        clearInterval(interval);
      }
  
      return () => clearInterval(interval);
    }, [gameStarted, gameEnded, timer]);

    const randomImage= () => {
      
      const randomIndex = Math.floor(Math.random() * (pImageArray.length + eImageArray.length - 1));
      
      
      let image;
      if (randomIndex >= pImageArray.length){
        image = eImageArray[randomIndex - pImageArray.length];  
      }
      else{
        image = pImageArray[randomIndex];
      }
      setDropImage(image);
    }

    const handleMove = (direction) => {
      if (direction === 'left') {
        setCatcherPosition((catcherPosition) => catcherPosition - 20);
      } else if (direction === 'right') {
        setCatcherPosition((catcherPosition) => catcherPosition + 20);
      }
    };

    
  
    useEffect(() => {

      if (gameEnded){
        return;
      }
      const handleKeyDown = (event) => {
      const catcherRect = document.querySelector('.catcher').getBoundingClientRect();
        if (event.key === 'ArrowLeft' && catcherRect.left - 20 > 0 ) {
          handleMove('left');
        } else if (event.key === 'ArrowRight' && catcherRect.right + 20 < screenWidth) {
          handleMove('right');
        }
      };
  
      window.addEventListener('keydown', handleKeyDown);
  
      return () => {
        window.removeEventListener('keydown', handleKeyDown);
      };
    });
  
    const startGame = () => {
      setGameStarted(true);
    };
  
    const endGame = () => {
      setGameEnded(true);
    };
  

    const resetDrop = (dropElement) => {
      dropElement.classList.add('no-transition');
      setDropPosition(-100); // Move the p4 image off-screen to visually remove it
      setTimeout(() => {
        dropElement.classList.remove('no-transition');
        randomImage();
      }, 30);
    }

    const dropAction = () => {
      const newDropPosition = dropPosition + dropUnit;

      // Check if the p4 image touches the catcher image
      const catcherRect = document.querySelector('.catcher').getBoundingClientRect();
      const dropRect = document.querySelector('.drop_item').getBoundingClientRect();
      const dropElement = document.querySelector('.drop_item');

      
      const isDropImageVisible = dropRect.bottom < screenHeight;
      

      if (dropRect.bottom >= catcherRect.top && dropRect.top <= catcherRect.bottom && dropRect.right >= catcherRect.left && dropRect.left <= catcherRect.right) {
        // Collision occurred, update score and remove the p4 image
        handleCatchItem(dropImage);
        resetDrop(dropElement); 
       
      } else if (isDropImageVisible) {
        // No collision, update the drop position
        setDropPosition(newDropPosition);
       
        
      }
      else{
        resetDrop(dropElement);
        
      }
      
    };

    const handleCatchItem = (dropImage) => {
      if (pImageArray.includes(dropImage)) {
        setScore((prevScore) => prevScore + 50);
      } else if (eImageArray.includes(dropImage)) {
        setScore((prevScore) => prevScore + 100);
      }
    };
  
    const changeUserName = (event) => {
      setUserName(event.target.value);
    };

    const handleSubmit = (event) => {
      event.preventDefault();
      
      // Post to backend
      fetch('http://localhost:8080/gamerecord', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          "userName" : userName,
          "score": score,
        }),
      }).then((response) => response.json())
      .then((responseData) => {
        console.log(responseData);
      })
      .catch((error) => {
        console.log(error);
      })
    };

    return (
      <div className="Game">
        
  
        {gameStarted && !gameEnded && (
          <div>
            <h2>Score: {score}</h2>
            <h2>Time Left: {Math.ceil(timer)}</h2>
            {/* Game elements */
              <div className="catcher_container">
                <img className="catcher" src={boat} alt="catcher" style={{ transform: `translateX(${catcherPosition}px)` }} />
              </div>
            }
            {<div className="drop_container">
              <img className="drop_item" src={dropImage} alt="drop_item" style={{ transform: `translateY(${dropPosition}%)`, transition: `transform 0.2s ease` }}></img>
              </div>}
          </div>
        )}
  
        {gameEnded && (
          <div>
            <h1>Game Over!</h1>
            <h2>Final Score: {score}</h2>
            <input type="text" placeholder="Enter your name" value = {userName} onChange={changeUserName}/>
            <button onClick={handleSubmit}>Submit</button>
            {/* Show ranking */}
          </div>
        )}
      </div>
    );
}

export default Game;