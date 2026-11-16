(** The kind of an invalid expression: a name read from the message that the
    term keeps. The text of a term writes the kind after [invalid], and nothing
    else of the expression.

    For a constructor of [Flambda.Invalid.t] other than [Message], the message
    is what [Flambda.Invalid.to_string] made: the name of the constructor alone
    if it has no argument, and "(" followed by the name if it has. [of_message]
    reads the name back, so the kind is then the constructor that the expression
    was made from. The text of a [Message] is kept as it is. If it has one of
    those two forms, the kind is the name in it, which may be the name of
    another constructor or a name that no constructor has. The word [Message]
    alone is recognised as a name too. A message of any other form has no name:
    [None].

    [Flambda2_compare.Compare.kind_of_invalid_message] is the same function. *)
val of_message : string -> string option

(** Whether a word can be written as a kind: [of_message] reads it back as
    itself. The printer writes a kind as it writes any name, in backquotes if
    the word is a keyword of the text. *)
val is_name : string -> bool
