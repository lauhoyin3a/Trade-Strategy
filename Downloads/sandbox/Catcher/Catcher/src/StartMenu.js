import './StartMenu.css';
import Game from './Game';
import LeaderBoard from './LeaderBoard';
import { useState } from 'react';
const StartMenu = () => {
    
    const [mode, setMode] = useState('default');

    const startGame = () => {
        setMode('game');
    }

    const visitBoard = () => {
        setMode('board');
    }

    return <div className='mode'>
        {mode === 'default' && (
        <div> <button onClick={startGame}>Start Game</button>
            <button onClick={visitBoard}>LeaderBoard</button>
            </div>
            )}
        {mode === 'game' && <Game/>}
        {mode === 'board' && <LeaderBoard/>}

    </div>
};

export default StartMenu;