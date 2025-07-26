# Custom Linux Shell

A simple yet feature-rich command-line shell implemented in C, demonstrating Linux system programming skills.

## Features

### 🚀 **Core Functionality**
- **Process Management**: Uses `fork()`, `execvp()`, and `waitpid()` for command execution
- **Built-in Commands**: `cd`, `pwd`, `exit`, `help`, `history`
- **Command Parsing**: Robust tokenization with support for multiple delimiters
- **Error Handling**: Comprehensive error reporting using `perror()`

### 🎨 **User Experience**
- **Colored Prompt**: Dynamic prompt showing `user@current_directory$`
- **Command History**: Tracks last 10 commands
- **Path Truncation**: Smart handling of long directory paths
- **Welcome Message**: Professional startup interface

### 🔧 **Technical Highlights**
- **POSIX Compliance**: Uses standard POSIX system calls
- **Memory Safe**: Proper buffer management and bounds checking
- **Modular Design**: Clean separation of concerns with well-documented functions
- **Signal Handling**: Graceful handling of EOF (Ctrl+D)

## Building and Running

### Prerequisites
- GCC compiler
- Linux/Unix environment
- Make utility

### Quick Start
```bash
# Clone and build
make

# Run the shell
make run

# Or run directly
./cmdshell
```

### Advanced Usage
```bash
# Debug build with symbols
make debug

# Install system-wide
make install

# Clean build artifacts
make clean

# View all available targets
make help
```

## Supported Commands

### Built-in Commands
| Command | Description | Example |
|---------|-------------|---------|
| `cd [dir]` | Change directory | `cd /home/user` |
| `pwd` | Print working directory | `pwd` |
| `exit` | Exit the shell | `exit` |
| `help` | Show available commands | `help` |
| `history` | Show command history | `history` |

### External Commands
All standard Linux commands available in `$PATH`:
- `ls`, `cat`, `grep`, `find`, `ps`, etc.
- Pipes and redirections (planned for future versions)

## Architecture

```
┌─────────────────────────────────────────┐
│               Main Loop                 │
├─────────────────────────────────────────┤
│  1. Display Prompt                      │
│  2. Read User Input                     │
│  3. Parse Command Line                  │
│  4. Execute Command                     │
│  5. Wait for Completion                 │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Command Execution             │
├─────────────────────────────────────────┤
│  Built-in Commands → Direct execution  │
│  External Commands → fork() + execvp() │
└─────────────────────────────────────────┘
```

## Code Structure

```c
// Key Functions
parse_line()      // Tokenize input into arguments
execute_command() // Handle built-ins and external commands
display_help()    // Show available commands
add_to_history()  // Track command history
main()           // Main shell loop
```

## Linux Skills Demonstrated

### 🔧 **System Programming**
- Process creation with `fork()`
- Program execution with `execvp()`
- Process synchronization with `waitpid()`
- Directory operations with `chdir()`, `getcwd()`

### 📚 **C Programming**
- Pointer manipulation and string handling
- Memory management and buffer safety
- Error handling and system call validation
- Modular function design

### 🛠️ **Development Tools**
- Makefile creation with multiple targets
- Compiler flags and debugging options
- Documentation with comments and README

### 🎯 **Best Practices**
- POSIX compliance for portability
- Defensive programming with bounds checking
- Clean code with proper documentation
- User-friendly interface design

## Future Enhancements

- [ ] Pipe support (`|`)
- [ ] I/O redirection (`>`, `<`, `>>`)
- [ ] Background processes (`&`)
- [ ] Tab completion
- [ ] Signal handling (Ctrl+C)
- [ ] Environment variable expansion
- [ ] Alias support
- [ ] Configuration file support

## Author

**Your Name**  
Demonstrating Linux system programming expertise through practical shell implementation.

---

*This project showcases fundamental Linux programming concepts and serves as a foundation for understanding how shells work at the system level.*
