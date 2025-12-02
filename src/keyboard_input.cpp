// keyboard_input.h
#include <windows.h>
#include <map>

class KeyboardController {
public:
    // VK_LEFT, VK_RIGHT, VK_SPACE (Salto), 'X' (Dash), 'Z' (Escalar)
    void press_key(int key_code) {
        INPUT ip;
        ip.type = INPUT_KEYBOARD;
        ip.ki.wScan = 0;
        ip.ki.time = 0;
        ip.ki.dwExtraInfo = 0;
        ip.ki.wVk = key_code;
        ip.ki.dwFlags = 0; // 0 significa presionar
        SendInput(1, &ip, sizeof(INPUT));
    }

    void release_key(int key_code) {
        INPUT ip;
        ip.type = INPUT_KEYBOARD;
        ip.ki.wScan = 0;
        ip.ki.time = 0;
        ip.ki.dwExtraInfo = 0;
        ip.ki.wVk = key_code;
        ip.ki.dwFlags = KEYEVENTF_KEYUP; // Soltar
        SendInput(1, &ip, sizeof(INPUT));
    }
    

    void update_key_state(int key_code, bool should_be_pressed) {
        // Un pequeño estado para no spamear Windows con "presionar, presionar, presionar"
        static std::map<int, bool> current_state;
        
        if (should_be_pressed && !current_state[key_code]) {
            press_key(key_code);
            current_state[key_code] = true;
        } else if (!should_be_pressed && current_state[key_code]) {
            release_key(key_code);
            current_state[key_code] = false;
        }
    }
};