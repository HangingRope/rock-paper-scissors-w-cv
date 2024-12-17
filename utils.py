import math

def get_gesture(lmList):
    """
    Get the gesture based on hand landmarks.

    Args:
        lmList (list): List of hand landmarks.

    Returns:
        str: Gesture (Rock, paper, or scissors).
    """
    distance_between_index_finger_and_wrist = int(math.hypot(lmList[0][1] - lmList[8][1], lmList[0][2] - lmList[8][2]))
    distance_between_ring_finger_and_wrist = int(math.hypot(lmList[16][1] - lmList[0][1], lmList[16][2] - lmList[0][2]))
    if (distance_between_index_finger_and_wrist > 250) and (distance_between_ring_finger_and_wrist > 250):
        return "Paper"
    elif (distance_between_index_finger_and_wrist > 250) and (distance_between_ring_finger_and_wrist < 250):
        return "Scissors"
    else:
        return "Rock"

def puttext(user_action, computer_action):
    """
    Determine the result message based on user and computer actions.

    Args:
        user_action (str): User's action (Rock, paper, or scissors).
        computer_action (str): Computer's action (Rock, paper, or scissors).

    Returns:
        str: Result message.
    """
    if user_action == computer_action:
        return "It's a tie!"
    elif user_action == "Rock":
        if computer_action == "Scissors":
            return "You win!"
        else:
            return "You lose."
    elif user_action == "Paper":
        if computer_action == "Rock":
            return "You win!"
        else:
            return "You lose."
    elif user_action == "Scissors":
        if computer_action == "Paper":
            return "You win!"
        else:
            return "You lose."
