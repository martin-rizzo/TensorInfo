

enum class Command {
    PRINT_METADATA,
    PRINT_TENSOR,
    LIST_TENSORS,
    EXTRACT_THUMBNAIL
};

struct Args {
    Command command  = Command::LIST_TENSORS;
    bool    no_color = false;

    Args( int argc, char* argv[] ) {
        
    }
};

