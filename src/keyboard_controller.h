#ifndef KEYBOARD_CONTROLLER_H
#define KEYBOARD_CONTROLLER_H

#include <windows.h>
#include <map>

class KeyboardController {
private:
    std::map<int, bool> key_state;

public:
    // Presiona o suelta una tecla según el estado 'pressed'
    void update_key(int key_code, bool pressed) {
        // Solo enviamos la señal si el estado cambia para no saturar
        if (key_state[key_code] == pressed) return;

        INPUT ip;
        ip.type = INPUT_KEYBOARD;
        ip.ki.wScan = 0;
        ip.ki.time = 0;
        ip.ki.dwExtraInfo = 0;
        ip.ki.wVk = key_code;
        
        if (pressed) {
            ip.ki.dwFlags = 0; // 0 = Presionar
        } else {
            ip.ki.dwFlags = KEYEVENTF_KEYUP; // Soltar
        }

        SendInput(1, &ip, sizeof(INPUT));
        key_state[key_code] = pressed;
    }
};

#endif