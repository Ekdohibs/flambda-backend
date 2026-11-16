(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* A term of the Rocq model of Flambda 2
   (middle_end/flambda2/rocq/FlambdaModel.v) written back as fexpr.

   This is the other direction of the conversion of flsemantics.ml, and it is
   written from the model and from the syntax of fexpr, not from that
   conversion: what a term of the model means is read off FlambdaModel.v and
   FlambdaPrimitiveModel.v, and the Flambda term that means the same is written
   down. The two are then checked against each other (flroundtrip.ml): a unit
   converted to the model, written back here, read again by the compiler's
   parser and converted again should be the term of the model it was.

   The result is a function of the term, of the two continuations of its unit,
   and of the kinds of the value slots, which the model keeps beside the term.
   Names are made from the numbers of the model: a variable [v12], a
   continuation [k3], a code id [code7], a value slot [vs2], a function slot
   [fs1], a symbol [$s5]. The two continuations of the unit are [done] and
   [error].

   What the model has not got is given the value that says least: no
   continuation sort but that of an exception handler, no inlining attribute,
   size 0 for code, code that is not recursive, allocation on the heap, the mode
   of the heap for an assignment, no subkinds. Code gets a region variable and a
   depth variable of its own, as the syntax asks for them; a call or an
   allocation in code names that region.

   What is covered: the terms that the conversion makes today. A term that is
   outside them, or that fexpr has no way to write, raises [Not_printable] with
   the reason: - a constant that is a vector or a mask, which the conversion
   does not make; - [Closure_of_set] other than as FlambdaModel.let_closures
   makes it: the closure at the first function slot of its set, followed by the
   projections of the other function slots, in order, each bound at kind
   [Value]. Flambda binds all the closures of a set at once; -
   [Static_const_closure_of_set] other than as FlambdaModel.bind_closures makes
   it: the closures of one static set next to each other, in the order of the
   set's symbols, of which there is at least one; - a closure bound at another
   kind than [Value]; - code whose parameters are not as many as the kinds of
   its arity; - a static boxed number whose contents are not a number that can
   be boxed, or are a variable whose binder is not in sight (the model does not
   say which number a boxed number holds; it is read off the kind of the
   variable); - a value slot of kind [Region] or [Rec_info]; - in a unit that
   binds tupled code, a direct call of that code, and an indirect call at other
   arities than those that the parser gives a call written without any. A call
   of the model gives tupled code a tuple. A direct call of fexpr, and an
   indirect call written with its arities, which is of known arity, give code
   its arguments as the code takes them: neither is the model's call. So in such
   a unit an indirect call is written without its arities, as a call of unknown
   arity, and the model's direct call of tupled code is not written at all.

   What fexpr can write and its parser reads differently is written all the
   same, so that the check shows it: the arity of the arguments of a direct
   call, the flag of [Is_int]. The kind of an [Opaque_identity] has no place in
   the text; the round trip gives each one of a unit that it reads the kind of
   its argument (flroundtrip.ml).

   A poison constant is written with a name of its own, in quotes. The model
   keeps no name for it, and the printer of fexpr writes a name as it is where
   the parser reads a string, so that a name without its quotes is not read
   back. A poison constant of kind naked [int8], naked [int16] or [mask] is
   written all the same: the parser of fexpr has no rule for those three kinds
   (parser/flambda_parser.mly, the rule [naked_number_kind]), though its printer
   writes them, so such a constant is not read back today, and the check shows
   it. *)

open Import
open Flambda2_kinds
open Flambda2_numbers
open Flambda2_bound_identifiers
module Kinds = Flambda2_rocq_semantics.FlambdaKindModel
module Primitives = Flambda2_rocq_semantics.FlambdaPrimitiveModel
module Model = Flambda2_rocq_semantics.FlambdaModel
module Numbers = Flambda2_rocq_semantics.BinNums
module P = Flambda_primitive
module F = Fexpr

exception Not_printable of string

let not_printable fmt =
  Format.kasprintf (fun reason -> raise (Not_printable reason)) fmt

(* ---- Numbers ---------------------------------------------------------- *)

let rec int_of_positive (p : Numbers.positive) =
  match p with
  | Coq_xH -> 1
  | Coq_xO p -> 2 * int_of_positive p
  | Coq_xI p -> (2 * int_of_positive p) + 1

let rec int64_of_positive (p : Numbers.positive) =
  match p with
  | Coq_xH -> 1L
  | Coq_xO p -> Int64.mul 2L (int64_of_positive p)
  | Coq_xI p -> Int64.add (Int64.mul 2L (int64_of_positive p)) 1L

let int64_of_z (z : Numbers.coq_Z) =
  match z with
  | Z0 -> 0L
  | Zpos p -> int64_of_positive p
  | Zneg p -> Int64.neg (int64_of_positive p)

let int_of_z z = Int64.to_int (int64_of_z z)

(* ---- Names ------------------------------------------------------------ *)

let located txt : _ F.located = Flambda_to_fexpr_commons.nowhere txt

let variable_prefix = "v"

let continuation_prefix = "k"

let code_id_prefix = "code"

let value_slot_prefix = "vs"

let function_slot_prefix = "fs"

let symbol_prefix = "s"

let name prefix number = prefix ^ string_of_int (int_of_positive number)

(* For showing where two terms differ, variables and continuations can be named
   by the order in which they are met instead of by their numbers: two terms
   that differ by the names of bound variables and continuations only are then
   printed the same. *)
let order_met : ((int, int) Hashtbl.t * (int, int) Hashtbl.t) option ref =
  ref None

let naming_by_order_met print =
  order_met := Some (Hashtbl.create 100, Hashtbl.create 100);
  Fun.protect ~finally:(fun () -> order_met := None) print

let met table number =
  let number = int_of_positive number in
  match Hashtbl.find_opt table number with
  | Some index -> index
  | None ->
    let index = Hashtbl.length table + 1 in
    Hashtbl.add table number index;
    index

let variable x : F.variable =
  match !order_met with
  | None -> located (name variable_prefix x)
  | Some (variables, _) ->
    located (variable_prefix ^ string_of_int (met variables x))

let continuation_id k : F.continuation_id =
  match !order_met with
  | None -> located (name continuation_prefix k)
  | Some (_, continuations) ->
    located (continuation_prefix ^ string_of_int (met continuations k))

let code_id id : F.code_id = located (name code_id_prefix id)

let value_slot slot : F.value_slot = located (name value_slot_prefix slot)

let function_slot slot : F.function_slot =
  located (name function_slot_prefix slot)

let symbol s : F.symbol = located (None, name symbol_prefix s)

(* The number in a name made as above: [Some n] for exactly the name that [name]
   makes of the number [n], and [None] for any other, such as one whose digits
   are not those that [string_of_int] writes. *)
let number_of_name ~prefix name =
  let prefix_length = String.length prefix in
  if
    String.length name > prefix_length
    && String.equal (String.sub name 0 prefix_length) prefix
  then
    match
      int_of_string_opt
        (String.sub name prefix_length (String.length name - prefix_length))
    with
    | Some n when n >= 1 && String.equal name (prefix ^ string_of_int n) ->
      Some n
    | Some _ | None -> None
  else None

(* The function slot that a projection of a value slot is given where the model
   does not name one (FlambdaModel.Project_value_slot). *)
let unnamed_function_slot : F.function_slot = located "unnamed"

(* The region and the depth variable that the syntax asks of code. *)
let region_of_code id : F.variable = located (name "region_of_code" id)

let depth_of_code id : F.variable = located (name "depth_of_code" id)

(* ---- Kinds, arities and constants ------------------------------------- *)

let naked_number_kind (kind : Kinds.naked_number_kind) :
    Flambda_kind.Naked_number_kind.t =
  match kind with
  | Kind_naked_immediate -> Naked_immediate
  | Kind_naked_float32 -> Naked_float32
  | Kind_naked_float -> Naked_float
  | Kind_naked_int8 -> Naked_int8
  | Kind_naked_int16 -> Naked_int16
  | Kind_naked_int32 -> Naked_int32
  | Kind_naked_int64 -> Naked_int64
  | Kind_naked_nativeint -> Naked_nativeint
  | Kind_naked_vec128 -> Naked_vec128
  | Kind_naked_vec256 -> Naked_vec256
  | Kind_naked_vec512 -> Naked_vec512
  | Kind_naked_mask -> Naked_mask

let kind (kind : Kinds.flambda_kind) : F.kind_with_subkind =
  match kind with
  | Value -> Value Anything
  | Naked_number kind -> Naked_number (naked_number_kind kind)
  | Region -> Region
  | Rec_info -> Rec_info

let compiler_kind (kind : Kinds.flambda_kind) : Flambda_kind.t =
  match kind with
  | Value -> Flambda_kind.value
  | Naked_number kind -> Flambda_kind.naked_number (naked_number_kind kind)
  | Region -> Flambda_kind.region
  | Rec_info -> Flambda_kind.rec_info

let rec arity_component (component : Model.arity_component) :
    F.kind_with_subkind =
  match component with
  | Singleton k -> kind k
  | Unboxed_product components ->
    Unboxed_product (List.map arity_component components)

let arity (arity : Model.arity) : F.arity = List.map arity_component arity

let const (c : Model.const) : F.const =
  match c with
  | Naked_immediate i -> Naked_immediate (Int64.to_string (int64_of_z i))
  | Tagged_immediate i -> Tagged_immediate (Int64.to_string (int64_of_z i))
  | Naked_float32 bits ->
    Naked_float32 (Int32.float_of_bits (Int64.to_int32 (int64_of_z bits)))
  | Naked_float bits -> Naked_float (Int64.float_of_bits (int64_of_z bits))
  | Naked_int8 i -> Naked_int8 (Numeric_types.Int8.of_int (int_of_z i))
  | Naked_int16 i -> Naked_int16 (Numeric_types.Int16.of_int (int_of_z i))
  | Naked_int32 i -> Naked_int32 (Int64.to_int32 (int64_of_z i))
  | Naked_int64 i -> Naked_int64 (int64_of_z i)
  | Naked_nativeint i -> Naked_nativeint (int64_of_z i)
  | Null -> Null
  | Naked_vec128 _ | Naked_vec256 _ | Naked_vec512 _ | Naked_mask _ ->
    not_printable "a constant that is a vector or a mask"
  | Poison k ->
    (* The printer of fexpr writes the name of a poison constant as it is, and
       the parser asks for a string there (parser/print_fexpr.ml, the case
       [Poison]; parser/flambda_parser.mly, the rule [const]). So the name is
       given with its quotes, and the text can be read. The model keeps no name:
       the conversion drops it. *)
    Poison (kind k, "\"poison\"")

let simple (s : Model.simple) : F.simple =
  match s with
  | Var x -> Var (variable x)
  | Symbol s -> Symbol (symbol s)
  | Const c -> Const (const c)

let simples = List.map simple

(* ---- Primitives ------------------------------------------------------- *)

(* A primitive of the model is written as the primitive of the compiler that
   means the same, and that one is given its syntax by the compiler's own
   description of the primitives of fexpr. *)

let machine_width = Target_system.Machine_width.Sixty_four

let standard_int (kind : Kinds.standard_int) : Flambda_kind.Standard_int.t =
  match kind with
  | Standard_tagged_immediate -> Tagged_immediate
  | Standard_naked_immediate -> Naked_immediate
  | Standard_naked_int8 -> Naked_int8
  | Standard_naked_int16 -> Naked_int16
  | Standard_naked_int32 -> Naked_int32
  | Standard_naked_int64 -> Naked_int64
  | Standard_naked_nativeint -> Naked_nativeint

let boxable_number (kind : Kinds.boxable_number) : Flambda_kind.Boxable_number.t
    =
  match kind with
  | Boxable_naked_float32 -> Naked_float32
  | Boxable_naked_float -> Naked_float
  | Boxable_naked_int32 -> Naked_int32
  | Boxable_naked_int64 -> Naked_int64
  | Boxable_naked_nativeint -> Naked_nativeint
  | Boxable_naked_vec128 -> Naked_vec128
  | Boxable_naked_vec256 -> Naked_vec256
  | Boxable_naked_vec512 -> Naked_vec512
  | Boxable_naked_mask -> Naked_mask

let signed_or_unsigned (signedness : Primitives.signedness) :
    P.signed_or_unsigned =
  match signedness with Signed -> Signed | Unsigned -> Unsigned

let scalar_signedness (signedness : Primitives.signedness) : Scalar.Signedness.t
    =
  match signedness with Signed -> Signed | Unsigned -> Unsigned

let or_unknown of_int (known : Numbers.coq_Z option) :
    _ Flambda2_lattices.Or_unknown.t =
  match known with Some n -> Known (of_int (int_of_z n)) | None -> Unknown

(* An allocation is on the heap; the region it names is put in afterwards
   ([in_region]). *)
let toplevel_region =
  lazy (Variable.create "toplevel_region" Flambda_kind.region)

let on_the_heap () =
  Alloc_mode.For_allocations.heap ~alloc_region:(Lazy.force toplevel_region)

let standard_int_or_float (kind : Kinds.standard_int) :
    Flambda_kind.Standard_int_or_float.t =
  Flambda_kind.Standard_int_or_float.of_standard_int (standard_int kind)

let array_kind (kind : Primitives.array_kind) : P.Array_kind.t =
  match kind with
  | Array_immediates -> Immediates
  | Array_gc_ignorable_values -> Gc_ignorable_values
  | Array_values -> Values

let unary_primitive (p : Primitives.unary_primitive) : P.unary_primitive =
  match p with
  | Block_load (Block_access_values (tag, size, field_kind), mut, field) ->
    let field_kind : P.Block_access_field_kind.t =
      match field_kind with Any_value -> Any_value | Immediate -> Immediate
    in
    let mut : Mutability.t =
      match mut with
      | Mutable -> Mutable
      | Immutable -> Immutable
      | Immutable_unique -> Immutable_unique
    in
    Block_load
      { kind =
          Values
            { tag = or_unknown Tag.Scannable.create_exn tag;
              size = or_unknown (Target_ocaml_int.of_int machine_width) size;
              field_kind
            };
        mut;
        field = Target_ocaml_int.of_int machine_width (int_of_z field)
      }
  | Is_int variant_only -> Is_int { variant_only }
  | Is_null -> Is_null
  | Get_tag -> Get_tag
  | Array_length (Array_kind kind) ->
    Array_length (Array_kind (array_kind kind))
  | Array_length Float_array_opt_dynamic -> Array_length Float_array_opt_dynamic
  | String_length -> String_length String
  | Unary_int_arith (kind, op) ->
    let op : P.unary_int_arith_op =
      match op with Not -> Not | Swap_byte_endianness -> Swap_byte_endianness
    in
    Int_arith (standard_int kind, op)
  | Num_conv (src, dst) ->
    Num_conv
      { src = standard_int_or_float src; dst = standard_int_or_float dst }
  | Boolean_not -> Boolean_not
  | Opaque_identity kind ->
    Opaque_identity { middle_end_only = false; kind = compiler_kind kind }
  | Unbox_number kind -> Unbox_number (boxable_number kind)
  | Box_number kind -> Box_number (boxable_number kind, on_the_heap ())
  | Untag_immediate -> Untag_immediate
  | Tag_immediate -> Tag_immediate
  | Reinterpret_64_bit_word reinterpret ->
    let reinterpret : P.Reinterpret_64_bit_word.t =
      match reinterpret with
      | Tagged_int63_as_unboxed_int64 -> Tagged_int63_as_unboxed_int64
      | Unboxed_int64_as_tagged_int63 -> Unboxed_int64_as_tagged_int63
    in
    Reinterpret_64_bit_word reinterpret

let binary_primitive (p : Primitives.binary_primitive) : P.binary_primitive =
  match p with
  | Array_load (kind, load_kind, mut) ->
    let load_kind : P.Array_load_kind.t =
      match load_kind with
      | Load_immediates -> Immediates
      | Load_gc_ignorable_values -> Gc_ignorable_values
      | Load_values -> Values
    in
    let mut : Mutability.t =
      match mut with
      | Mutable -> Mutable
      | Immutable -> Immutable
      | Immutable_unique -> Immutable_unique
    in
    Array_load (array_kind kind, load_kind, mut)
  | String_or_bigstring_load width ->
    let width : P.string_accessor_width =
      match width with
      | Eight -> Eight
      | Eight_signed -> Eight_signed
      | Sixteen -> Sixteen
      | Sixteen_signed -> Sixteen_signed
      | Thirty_two -> Thirty_two
      | Sixty_four -> Sixty_four
    in
    String_or_bigstring_load (String, width)
  | Phys_equal op ->
    let op : P.equality_comparison =
      match op with Phys_eq -> Eq | Phys_neq -> Neq
    in
    Phys_equal op
  | Int_shift (kind, op) ->
    let op : P.int_shift_op =
      match op with Lsl -> Lsl | Lsr -> Lsr | Asr -> Asr
    in
    Int_shift (standard_int kind, op)
  | Int_arith (kind, op) ->
    let op : P.binary_int_arith_op =
      match op with
      | Add -> Add
      | Sub -> Sub
      | Mul -> Mul
      | Div signedness -> Div (scalar_signedness signedness)
      | Mod signedness -> Mod (scalar_signedness signedness)
      | And -> And
      | Or -> Or
      | Xor -> Xor
    in
    Int_arith (standard_int kind, op)
  | Int_comp (kind, behaviour) ->
    let behaviour : P.signed_or_unsigned P.comparison_behaviour =
      match behaviour with
      | Yielding_bool Int_eq -> Yielding_bool Eq
      | Yielding_bool Int_neq -> Yielding_bool Neq
      | Yielding_bool (Int_lt signedness) ->
        Yielding_bool (Lt (signed_or_unsigned signedness))
      | Yielding_bool (Int_gt signedness) ->
        Yielding_bool (Gt (signed_or_unsigned signedness))
      | Yielding_bool (Int_le signedness) ->
        Yielding_bool (Le (signed_or_unsigned signedness))
      | Yielding_bool (Int_ge signedness) ->
        Yielding_bool (Ge (signed_or_unsigned signedness))
      | Yielding_int_like_compare_functions signedness ->
        Yielding_int_like_compare_functions (signed_or_unsigned signedness)
    in
    Int_comp (standard_int kind, behaviour)

let variadic_primitive (p : Primitives.variadic_primitive) ~num_args :
    P.variadic_primitive =
  match p with
  | Make_block (Block_kind_values tag) ->
    Make_block
      ( Values
          ( Tag.Scannable.create_exn (int_of_z tag),
            List.init num_args (fun _ -> Flambda_kind.With_subkind.any_value) ),
        Immutable,
        on_the_heap () )
  | Make_immutable_array kind ->
    Make_array (array_kind kind, Immutable, on_the_heap ())

(* A primitive of the heap is one of three sorts of primitive of the compiler,
   by the number of its arguments. An assignment is written with the mode of the
   heap: the model's store has no mode. *)
type heap_primitive =
  | Heap_binary of P.binary_primitive
  | Heap_ternary of P.ternary_primitive
  | Heap_variadic of P.variadic_primitive

let init_or_assign (init : Primitives.init_or_assign) : P.Init_or_assign.t =
  match init with
  | Initialization -> Initialization
  | Assignment -> Assignment Alloc_mode.For_assignments.heap

let heap_primitive (p : Primitives.heap_primitive) ~num_args : heap_primitive =
  match p with
  | Make_block_with_identity (mut, tag) ->
    let mut : Mutability.t =
      match mut with
      | Cell_mutable -> Mutable
      | Cell_immutable_unique -> Immutable_unique
    in
    Heap_variadic
      (Make_block
         ( Values
             ( Tag.Scannable.create_exn (int_of_z tag),
               List.init num_args (fun _ -> Flambda_kind.With_subkind.any_value)
             ),
           mut,
           on_the_heap () ))
  | Block_set (Block_access_values (tag, size, field_kind), init, field) ->
    let field_kind : P.Block_access_field_kind.t =
      match field_kind with Any_value -> Any_value | Immediate -> Immediate
    in
    Heap_binary
      (Block_set
         { kind =
             Values
               { tag = or_unknown Tag.Scannable.create_exn tag;
                 size = or_unknown (Target_ocaml_int.of_int machine_width) size;
                 field_kind
               };
           init = init_or_assign init;
           field = Target_ocaml_int.of_int machine_width (int_of_z field)
         })
  | Make_mutable_array kind ->
    Heap_variadic (Make_array (array_kind kind, Mutable, on_the_heap ()))
  | Array_set (kind, set_kind) ->
    let set_kind : P.Array_set_kind.t =
      match set_kind with
      | Set_immediates -> Immediates
      | Set_gc_ignorable_values -> Gc_ignorable_values
      | Set_values init -> Values (init_or_assign init)
    in
    Heap_ternary (Array_set (array_kind kind, set_kind))

(* The region that the compiler's description wrote as the toplevel one is the
   region of where the primitive is. *)
let rec in_region (region : F.region) (param : F.prim_param) : F.prim_param =
  match region, param with
  | Toplevel_alloc_region, _ -> param
  | Named region, Labeled ({ txt = "toplevel"; loc }, []) ->
    Labeled ({ txt = region.txt; loc }, [])
  | Named _, Labeled (label, params) ->
    Labeled (label, List.map (in_region region) params)
  | Named _, Anonymous params -> Anonymous (List.map (in_region region) params)

let encoding =
  lazy
    (Flambda_to_fexpr_commons.Env.bind_toplevel_alloc_region
       (Flambda_to_fexpr_commons.Env.create ())
       (Lazy.force toplevel_region))

let prim region (p : Primitives.prim) (args : Model.simple list) : F.prim =
  let encoding = Lazy.force encoding in
  let op : F.prim_op =
    match p with
    | Unary p -> Fexpr_prim.OfFlambda.unop encoding (unary_primitive p)
    | Binary p -> Fexpr_prim.OfFlambda.binop encoding (binary_primitive p)
    | Variadic p ->
      Fexpr_prim.OfFlambda.varop encoding
        (variadic_primitive p ~num_args:(List.length args))
    | Heap p -> (
      match heap_primitive p ~num_args:(List.length args) with
      | Heap_binary p -> Fexpr_prim.OfFlambda.binop encoding p
      | Heap_ternary p -> Fexpr_prim.OfFlambda.ternop encoding p
      | Heap_variadic p -> Fexpr_prim.OfFlambda.varop encoding p)
    | Region_primitive _ ->
      not_printable
        "a primitive of regions, which the conversion makes for the machine of \
         regions only"
  in
  { op with params = List.map (in_region region) op.params }, simples args

(* The two projections name slots, and are written here. *)
let positional_name (name : string F.located) : F.prim_param =
  Anonymous [Labeled (name, [])]

let project_value_slot ~project_from slot closure : F.prim =
  ( { prim = "%project_value_slot";
      params = [positional_name project_from; positional_name (value_slot slot)]
    },
    [simple closure] )

let project_function_slot ~move_from ~move_to closure : F.prim =
  ( { prim = "%project_function_slot";
      params =
        [ positional_name (function_slot move_from);
          positional_name (function_slot move_to) ]
    },
    [simple closure] )

(* ---- Terms ------------------------------------------------------------ *)

type env =
  { unit_return : Model.continuation;
    unit_exn : Model.continuation;
    (* Where a call or an allocation is: at toplevel, or in code. *)
    region : F.region;
    (* The variables in scope, with the kinds of their binders. *)
    kinds : (Model.variable * Kinds.flambda_kind) list;
    (* The kinds of the value slots, which the model keeps beside the term. *)
    value_slot_kind : Model.value_slot -> Kinds.flambda_kind option;
    (* The code of the unit that is tupled. *)
    tupled_code : Model.code_id list
  }

let bind env x kind = { env with kinds = (x, kind) :: env.kinds }

let continuation env k : F.continuation =
  if k = env.unit_return
  then Special Done
  else if k = env.unit_exn
  then Special Error
  else Named (continuation_id k)

let kinded_parameters params kinds : F.kinded_parameter list =
  List.map2
    (fun param k : F.kinded_parameter ->
      { param = variable param; kind = Some (kind k) })
    params kinds

let bind_parameters env params kinds = List.fold_left2 bind env params kinds

let value_slots env (slots : (Model.value_slot * Model.simple) list) :
    F.value_slots option =
  match slots with
  | [] -> None
  | _ :: _ ->
    Some
      (List.map
         (fun (slot, s) : F.one_value_slot ->
           let kind =
             match env.value_slot_kind slot with
             | None | Some Value -> None
             | Some (Naked_number kind) -> Some (naked_number_kind kind)
             | Some (Region | Rec_info) ->
               not_printable "a value slot of kind region or rec_info"
           in
           { var = value_slot slot; value = simple s; kind })
         slots)

let on_the_heap_in env : F.alloc_mode_for_allocations =
  Heap { alloc_region = env.region }

let raise_kind (kind : Model.raise_kind) : F.raise_kind =
  match kind with
  | Regular -> Regular
  | Reraise -> Reraise
  | No_trace -> No_trace

let trap_action env (action : Model.trap_action) : F.trap_action =
  match action with
  | Push exn_handler -> Push { exn_handler = continuation env exn_handler }
  | Pop (exn_handler, kind) ->
    Pop
      { exn_handler = continuation env exn_handler;
        raise_kind = Option.map raise_kind kind
      }

let apply_cont env (ac : Model.apply_cont) : F.apply_cont =
  { cont = continuation env ac.ac_continuation;
    trap_action = Option.map (trap_action env) ac.ac_trap_action;
    args = simples ac.ac_args
  }

(* The projections that FlambdaModel.let_closures puts after the closure at the
   first function slot of a set: one for each other function of the set, in
   order, from the variable [base] bound to the first. *)
let rec projections_of_set ~base ~base_slot other_decls (body : Model.expr) =
  match other_decls, body with
  | [], _ -> [], body
  | ( (slot, code) :: other_decls,
      Let
        ( x,
          Value,
          Project_function_slot (move_from, move_to, Var projected),
          body ) )
    when move_from = base_slot && move_to = slot && projected = base ->
    let bound, body = projections_of_set ~base ~base_slot other_decls body in
    (x, slot, code) :: bound, body
  | _ :: _, _ ->
    not_printable
      "a closure of a set of several functions that is not followed by the \
       projections of the others"
[@@ocaml.warning "-4"]

(* The closures of one static set, which FlambdaModel.bind_closures puts next to
   each other, one for each symbol of the set, in order. *)
let rec closures_of_static_set closure_symbols function_decls slots
    (bindings : Model.static_binding list) =
  match closure_symbols, bindings with
  | [], _ -> bindings
  | ( (slot, s) :: closure_symbols,
      Bind_symbol
        (s', Static_const_closure_of_set (slot', _, function_decls', slots'))
      :: bindings )
    when s = s' && slot = slot'
         && function_decls = function_decls'
         && slots = slots' ->
    closures_of_static_set closure_symbols function_decls slots bindings
  | _ :: _, _ ->
    not_printable
      "the closures of a static set that are not bound next to each other, in \
       the order of its symbols"
[@@ocaml.warning "-4"]

let static_field (s : Model.simple) : F.field_of_block =
  match s with
  | Var x -> Dynamically_computed (variable x)
  | Symbol s -> Symbol (symbol s)
  | Const c -> Const (const c)

let boxed_number env (contents : Model.simple) : F.static_data =
  let of_kind (kind : Kinds.flambda_kind option) x : F.static_data =
    match kind with
    | Some (Naked_number Kind_naked_float32) -> Boxed_float32 (Var x)
    | Some (Naked_number Kind_naked_float) -> Boxed_float (Var x)
    | Some (Naked_number Kind_naked_int32) -> Boxed_int32 (Var x)
    | Some (Naked_number Kind_naked_int64) -> Boxed_int64 (Var x)
    | Some (Naked_number Kind_naked_nativeint) -> Boxed_nativeint (Var x)
    | Some
        ( Value | Region | Rec_info
        | Naked_number
            ( Kind_naked_immediate | Kind_naked_int8 | Kind_naked_int16
            | Kind_naked_vec128 | Kind_naked_vec256 | Kind_naked_vec512
            | Kind_naked_mask ) ) ->
      not_printable
        "a static boxed number that holds a variable of a kind that is not \
         boxed"
    | None ->
      not_printable
        "a static boxed number that holds a variable whose binder is not in \
         sight"
  in
  match contents with
  | Const (Naked_float32 bits) ->
    Boxed_float32
      (Const (Int32.float_of_bits (Int64.to_int32 (int64_of_z bits))))
  | Const (Naked_float bits) ->
    Boxed_float (Const (Int64.float_of_bits (int64_of_z bits)))
  | Const (Naked_int32 i) -> Boxed_int32 (Const (Int64.to_int32 (int64_of_z i)))
  | Const (Naked_int64 i) -> Boxed_int64 (Const (int64_of_z i))
  | Const (Naked_nativeint i) -> Boxed_nativeint (Const (int64_of_z i))
  | Var x -> of_kind (List.assoc_opt x env.kinds) (variable x)
  | Symbol _
  | Const
      ( Naked_immediate _ | Tagged_immediate _ | Naked_int8 _ | Naked_int16 _
      | Naked_vec128 _ | Naked_vec256 _ | Naked_vec512 _ | Naked_mask _ | Null
      | Poison _ ) ->
    not_printable "a static boxed number that holds what is not boxed"

(* A function of a set: its code and, where the set names its function slots,
   its slot. *)
type function_of_set =
  { code_id : F.code_id;
    function_slot : F.function_slot option
  }

let rec expr env (e : Model.expr) : F.expr =
  match e with
  | Let (x, k, Simple s, body) -> let_ env x k (F.Simple (simple s)) body
  | Let (x, k, Prim (p, args), body) ->
    let_ env x k (F.Prim (prim env.region p args)) body
  | Let (x, k, Project_value_slot (slot, closure), body) ->
    let_ env x k
      (F.Prim
         (project_value_slot ~project_from:unnamed_function_slot slot closure))
      body
  | Let (x, k, Project_value_slot_from (project_from, slot, closure), body) ->
    let_ env x k
      (F.Prim
         (project_value_slot
            ~project_from:(function_slot project_from)
            slot closure))
      body
  | Let (x, k, Project_function_slot (move_from, move_to, closure), body) ->
    let_ env x k
      (F.Prim (project_function_slot ~move_from ~move_to closure))
      body
  | Let (x, k, Set_of_closures (code, slots), body) ->
    closures env
      [ ( x,
          k,
          ({ code_id = code_id code; function_slot = None } : function_of_set) )
      ]
      slots body
  | Let (x, k, Closure_of_set (slot, function_decls, slots), body) -> (
    match function_decls with
    | (first_slot, code) :: other_decls when first_slot = slot ->
      let others, body =
        projections_of_set ~base:x ~base_slot:slot other_decls body
      in
      closures env
        (( x,
           k,
           ({ code_id = code_id code;
              function_slot = Some (function_slot slot)
            }
             : function_of_set) )
        :: List.map
             (fun (x, slot, code) ->
               ( x,
                 Kinds.Value,
                 ({ code_id = code_id code;
                    function_slot = Some (function_slot slot)
                  }
                   : function_of_set) ))
             others)
        slots body
    | [] | _ :: _ ->
      not_printable
        "a closure of a set that is not at the first function slot of the set")
  | Let_static (bindings, body) ->
    F.Let_symbol
      { bindings = static_bindings env bindings;
        value_slots = None;
        body = expr env body
      }
  | Let_cont_non_recursive (k, h, body) ->
    F.Let_cont
      { recursive = Nonrecursive;
        body = expr env body;
        bindings = [handler env k h]
      }
  | Let_cont_recursive (handlers, body) ->
    (* The model has no parameters shared by the handlers of a group: each
       handler has all of its own. *)
    F.Let_cont
      { recursive = Recursive [];
        body = expr env body;
        bindings = List.map (fun (k, h) -> handler env k h) handlers
      }
  | Apply ap ->
    let params_arity = arity ap.ap_args_arity in
    let call_kind : F.function_call =
      match ap.ap_call_kind with
      | Direct code ->
        if List.mem code env.tupled_code
        then
          not_printable
            "a direct call of tupled code, which the model gives a tuple";
        Direct { code_id = code_id code; function_slot = None }
      | Indirect -> Indirect
    in
    (* In a unit that binds tupled code an indirect call is written without its
       arities: with them it would be read as a call of known arity. The parser
       then gives it one value for each argument and one value returned. *)
    let without_arities =
      match ap.ap_call_kind, env.tupled_code with
      | Indirect, _ :: _ ->
        let one_value : Model.arity = [Singleton Kinds.Value] in
        if
          ap.ap_args_arity
          <> List.map
               (fun _ : Model.arity_component -> Singleton Kinds.Value)
               ap.ap_args
          || ap.ap_return_arity <> one_value
        then
          not_printable
            "an indirect call, in a unit that binds tupled code, at other \
             arities than one value for each argument and one value returned";
        true
      | Indirect, [] | Direct _, _ -> false
    in
    F.Apply
      { func = Some (simple ap.ap_callee);
        continuation = Return (continuation env ap.ap_continuation);
        exn_continuation =
          ( continuation env ap.ap_exn_continuation,
            List.map (fun (s, k) -> simple s, kind k) ap.ap_exn_extra_args );
        args = simples ap.ap_args;
        call_kind = Function call_kind;
        alloc_mode = Not_alloc_stack { alloc_region = env.region };
        arities =
          (if without_arities
           then None
           else
             Some
               { params_arity = Some params_arity;
                 ret_arity = arity ap.ap_return_arity
               });
        inlined = None;
        inlining_state = None;
        nontail = false
      }
  | Apply_cont ac -> F.Apply_cont (apply_cont env ac)
  | Switch (scrutinee, arms) ->
    F.Switch
      { scrutinee = simple scrutinee;
        cases =
          List.map
            (fun (discriminant, ac) : (int * F.apply_or_inlined_cont) ->
              int_of_z discriminant, Named_cont (apply_cont env ac))
            arms
      }
  | Invalid -> F.Invalid

and let_ env x k (defining_expr : F.named) body : F.expr =
  F.Let
    { bindings = [{ var = variable x; defining_expr }];
      value_slots = None;
      body = expr (bind env x k) body
    }

(* The closures of one set, bound together. *)
and closures env bound slots body : F.expr =
  let env_of_body =
    List.fold_left
      (fun env (x, (k : Kinds.flambda_kind), _) ->
        (match k with
        | Value -> ()
        | Naked_number _ | Region | Rec_info ->
          not_printable "a closure bound at another kind than value");
        bind env x k)
      env bound
  in
  F.Let
    { bindings =
        List.map
          (fun (x, _, ({ code_id; function_slot } : function_of_set)) :
               F.let_binding ->
            { var = variable x;
              defining_expr =
                F.Closure { code_id; function_slot; alloc = on_the_heap_in env }
            })
          bound;
      value_slots = value_slots env slots;
      body = expr env_of_body body
    }

and handler env k (Model.Handler (params, metadata, handler)) :
    F.continuation_binding =
  let kinds = metadata.hm_params_kinds in
  if List.compare_lengths params kinds <> 0
  then
    not_printable
      "a handler whose parameters are not as many as the kinds it gives them";
  { name = continuation_id k;
    params = kinded_parameters params kinds;
    sort = (if metadata.hm_is_exn_handler then Some Exn else None);
    cold = false;
    handler = expr (bind_parameters env params kinds) handler
  }

and static_bindings env (bindings : Model.static_binding list) :
    F.symbol_binding list =
  match bindings with
  | [] -> []
  | Bind_code (id, c) :: bindings ->
    F.Code (code env id c) :: static_bindings env bindings
  | Bind_deleted_code id :: bindings ->
    F.Deleted_code (code_id id) :: static_bindings env bindings
  | Bind_symbol (s, Static_const_set_of_closures (code, slots)) :: bindings ->
    F.Set_of_closures
      { bindings =
          [ { symbol = symbol s;
              fun_decl =
                { code_id = code_id code;
                  function_slot = None;
                  alloc = on_the_heap_in env
                }
            } ];
        elements = value_slots env slots
      }
    :: static_bindings env bindings
  | Bind_symbol (_, Static_const_closure_of_set (_, [], _, _)) :: _ ->
    (* Without this case, no binding would be consumed below. *)
    not_printable "a static closure of a set that has no symbols"
  | Bind_symbol
      ( _,
        Static_const_closure_of_set (_, closure_symbols, function_decls, slots)
      )
    :: _ ->
    let bindings =
      closures_of_static_set closure_symbols function_decls slots bindings
    in
    F.Set_of_closures
      { bindings =
          List.map
            (fun (slot, s) : F.static_closure_binding ->
              let code =
                match List.assoc_opt slot function_decls with
                | Some code -> code
                | None ->
                  not_printable
                    "a static closure at a function slot that its set has not \
                     got"
              in
              { symbol = symbol s;
                fun_decl =
                  { code_id = code_id code;
                    function_slot = Some (function_slot slot);
                    alloc = on_the_heap_in env
                  }
              })
            closure_symbols;
        elements = value_slots env slots
      }
    :: static_bindings env bindings
  | Bind_symbol (s, Static_const_block (tag, fields)) :: bindings ->
    F.Data
      { symbol = symbol s;
        defining_expr =
          Block
            { tag = int_of_z tag;
              mutability = Immutable;
              elements = List.map static_field fields
            }
      }
    :: static_bindings env bindings
  | Bind_symbol (s, Static_const_boxed_number contents) :: bindings ->
    F.Data { symbol = symbol s; defining_expr = boxed_number env contents }
    :: static_bindings env bindings
  | Bind_symbol (s, Static_const_immutable_value_array elements) :: bindings ->
    F.Data
      { symbol = symbol s;
        defining_expr = Immutable_value_array (List.map static_field elements)
      }
    :: static_bindings env bindings
  | Bind_symbol (s, Static_const_immutable_string bytes) :: bindings ->
    F.Data
      { symbol = symbol s;
        defining_expr =
          Immutable_string
            (String.init (List.length bytes) (fun i ->
                 Char.chr (int_of_z (List.nth bytes i))))
      }
    :: static_bindings env bindings

and code env id (Model.Code (params, my_closure, ret, exn, metadata, body)) :
    F.code =
  let kinds = Model.unarize metadata.cm_params_arity in
  if List.compare_lengths params kinds <> 0
  then
    not_printable
      "code whose parameters are not as many as the kinds of its arity";
  let region = region_of_code id in
  (* Code is closed: its body sees its parameters and its closure only. *)
  let env_of_body =
    bind_parameters
      { env with region = Named region; kinds = [my_closure, Kinds.Value] }
      params kinds
  in
  { id = code_id id;
    newer_version_of = None;
    param_arity = Some (arity metadata.cm_params_arity);
    ret_arity = Some (arity metadata.cm_result_arity);
    recursive = Nonrecursive;
    inline = None;
    params_and_body =
      { params = kinded_parameters params kinds;
        closure_var = variable my_closure;
        region_vars = Not_alloc_stack { alloc_region = region };
        depth_var = depth_of_code id;
        ret_cont = continuation_id ret;
        exn_cont = continuation_id exn;
        body = expr env_of_body body
      };
    code_size = 0;
    is_tupled = metadata.cm_is_tupled;
    stub = metadata.cm_is_stub;
    is_a_functor = false;
    loopify = None;
    result_mode = Not_alloc_stack
  }

(* The code ids that a term binds to tupled code, wherever the binding is. *)
let rec tupled_code_of (e : Model.expr) : Model.code_id list =
  let of_handler (Model.Handler (_, _, handler)) = tupled_code_of handler in
  match e with
  | Let (_, _, _, body) -> tupled_code_of body
  | Let_static (bindings, body) ->
    List.concat_map
      (fun (binding : Model.static_binding) ->
        match binding with
        | Bind_code (id, Code (_, _, _, _, metadata, code_body)) ->
          (if metadata.cm_is_tupled then [id] else [])
          @ tupled_code_of code_body
        | Bind_deleted_code _ | Bind_symbol _ -> [])
      bindings
    @ tupled_code_of body
  | Let_cont_non_recursive (_, h, body) -> of_handler h @ tupled_code_of body
  | Let_cont_recursive (handlers, body) ->
    List.concat_map (fun (_, h) -> of_handler h) handlers @ tupled_code_of body
  | Apply _ | Apply_cont _ | Switch _ | Invalid -> []

(* A unit of the model: its body and its two continuations. *)
let unit ~value_slot_kind ~return_continuation ~exn_continuation
    (body : Model.expr) : F.flambda_unit =
  let env =
    { unit_return = return_continuation;
      unit_exn = exn_continuation;
      region = Toplevel_alloc_region;
      kinds = [];
      value_slot_kind;
      tupled_code = tupled_code_of body
    }
  in
  { body = expr env body }

let print ppf (unit : F.flambda_unit) =
  Format.fprintf ppf "%a@." Print_fexpr.flambda_unit unit

let to_string unit =
  Misc.Colours.without_colours ~f:(fun () -> Format.asprintf "%a" print unit)
