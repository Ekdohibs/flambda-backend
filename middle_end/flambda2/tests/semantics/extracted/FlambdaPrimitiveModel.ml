open BinNums
open FlambdaKindModel

type signedness =
  | Signed
  | Unsigned

type block_access_field_kind =
  | Any_value
  | Immediate

type block_access_kind =
  | Block_access_values of coq_Z option * coq_Z option * block_access_field_kind

type mutability =
  | Mutable
  | Immutable
  | Immutable_unique

type unary_int_arith_op =
  | Not
  | Swap_byte_endianness

type reinterpret_64_bit_word =
  | Tagged_int63_as_unboxed_int64
  | Unboxed_int64_as_tagged_int63

type array_kind =
  | Array_immediates
  | Array_gc_ignorable_values
  | Array_values

type array_kind_for_length =
  | Array_kind of array_kind
  | Float_array_opt_dynamic

type array_load_kind =
  | Load_immediates
  | Load_gc_ignorable_values
  | Load_values

type unary_primitive =
  | Block_load of block_access_kind * mutability * coq_Z
  | Is_int of bool
  | Is_null
  | Get_tag
  | Array_length of array_kind_for_length
  | String_length
  | Opaque_identity of flambda_kind
  | Unary_int_arith of standard_int * unary_int_arith_op
  | Num_conv of standard_int * standard_int
  | Boolean_not
  | Unbox_number of boxable_number
  | Box_number of boxable_number
  | Untag_immediate
  | Tag_immediate
  | Reinterpret_64_bit_word of reinterpret_64_bit_word

type binary_int_arith_op =
  | Add
  | Sub
  | Mul
  | Div of signedness
  | Mod of signedness
  | And
  | Or
  | Xor

type int_shift_op =
  | Lsl
  | Lsr
  | Asr

type int_comparison =
  | Int_eq
  | Int_neq
  | Int_lt of signedness
  | Int_gt of signedness
  | Int_le of signedness
  | Int_ge of signedness

type int_comparison_behaviour =
  | Yielding_bool of int_comparison
  | Yielding_int_like_compare_functions of signedness

type string_accessor_width =
  | Eight
  | Eight_signed
  | Sixteen
  | Sixteen_signed
  | Thirty_two
  | Sixty_four

type equality_comparison =
  | Phys_eq
  | Phys_neq

type binary_primitive =
  | Array_load of array_kind * array_load_kind * mutability
  | String_or_bigstring_load of string_accessor_width
  | Phys_equal of equality_comparison
  | Int_arith of standard_int * binary_int_arith_op
  | Int_shift of standard_int * int_shift_op
  | Int_comp of standard_int * int_comparison_behaviour

type block_kind = Block_kind_values of coq_Z

type variadic_primitive =
  | Make_block of block_kind
  | Make_immutable_array of array_kind

type cell_mutability =
  | Cell_mutable
  | Cell_immutable_unique

type init_or_assign =
  | Initialization
  | Assignment

type array_set_kind =
  | Set_immediates
  | Set_gc_ignorable_values
  | Set_values of init_or_assign

type heap_primitive =
  | Make_block_with_identity of cell_mutability * coq_Z
  | Block_set of block_access_kind * init_or_assign * coq_Z
  | Make_mutable_array of array_kind
  | Array_set of array_kind * array_set_kind

type region_primitive =
  | Begin_region
  | Begin_try_region
  | End_region
  | End_try_region
  | Local_allocation

type prim =
  | Unary of unary_primitive
  | Binary of binary_primitive
  | Variadic of variadic_primitive
  | Heap of heap_primitive
  | Region_primitive of region_primitive
