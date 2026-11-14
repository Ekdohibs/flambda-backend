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


(* The length and an element of an immutable array that is known: a
   normalisation that refines.

   This file is to be reviewed: [load_known_arrays] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; [rename_known_arrays]; and the
   statements.

   What Simplify does. It gives a [Make_array] that is marked
   [Immutable] the type of an immutable array of the types of its
   arguments
   ([Simplify_variadic_primitive.simplify_make_array]), and a symbol
   that is bound to an immutable array of values the type of its
   static constant. With such a type the length of the array is the
   number of its elements, a constant
   ([Simplify_unary_primitive.simplify_array_length], a projection of
   the type), and a load at an index that is one known tagged immediate
   has the type of that element, which is reified: the element itself
   where it is a constant, a symbol or a variable
   ([Simplify_binary_primitive.simplify_array_load], which does not
   look at the mutability that the load carries, and gives [Invalid]
   where the index is out of the bounds). So in the term after
   Simplify the length of an array that the same code has made is a
   constant, the test of an index against it is decided or is a test
   against a constant, a load at a constant index is the element, and
   an array that is then unused is gone.

   What is here. There are no types of arrays in the model, so
   [load_known_arrays] does it for the two cases in which an array is
   known from the term alone, as FlambdaBlockLoadSpec.load_known_blocks
   does for the fields of a block. It goes through a term keeping
   ([known_arrays]):

   - for a variable bound by a [Let] to a [Make_immutable_array], the
     arguments of the primitive ([array_made_by]). The primitive of the
     model is the one of an immutable array of values
     (FlambdaPrimitiveModel.v): an array that is mutable is made by a
     primitive of the heap, so nothing is ever known of one;
   - for a symbol bound by a [Let_static] to a
     [Static_const_immutable_value_array], its elements
     ([static_array_made]). That too is immutable.

   The two are read alike, through FlambdaValueSpec.array_of, by the
   primitives and by the proof.

   A [Let] of a primitive applied to such a variable or symbol becomes
   a [Let] of a [Simple] ([load_known_array_named], [of_known_array]):

   - [Array_length], at any array kind, becomes the number of the
     elements, a tagged immediate, if that number is one
     ([length_of_known]);
   - [Array_load], at any kinds and any mark of mutability, at an index
     that is a constant tagged immediate, at least 0 and below the
     number of the elements, becomes what the element holds, if that is
     still known ([element_loaded]).

   Anything else is left as it is: a load at an index that is a
   variable; and a load at a constant index out of the bounds, which
   has no value (FlambdaPrimitiveSpec.array_index) and which Simplify
   makes [Invalid]. So the pass does not fire there, although anything
   would refine the term.

   Nothing else is rewritten. The [Let] stays, as with
   FlambdaBlockLoadSpec.load_known_blocks: the variable is then a
   constant or an alias of the element, which other passes follow, and
   the [Make_immutable_array] is removed if it is then unused.

   Going under a binder, what is known of its name is forgotten, and so
   is what any element that is that name holds ([forget_array_var],
   with FlambdaBlockLoadSpec.forget_field): the name stands for
   something else from there on. The number of the elements of such an
   array stays known, and so does what its other elements hold. A
   handler of a non-recursive continuation is gone into, with what is
   known where it is defined; the handlers of recursive ones and the
   bodies of code are left as they are
   (FlambdaValidateReadingSpec.read_arrays applies the pass in them
   too, with nothing known there).

   The result refines the term ([load_known_arrays_refines_statement]).
   It is not claimed to be equivalent to it: a load has no value where
   the element is not one that its load kind holds
   (FlambdaPrimitiveSpec.array_loaded), and the element has one. So
   the pass is for the term before Simplify. The fold of the length
   keeps the range check of the primitive: where the number of the
   elements is not a tagged immediate, the primitive is left as it
   is.

   It is not a step of the rounds of
   FlambdaValidateSpec.validate_unit: FlambdaValidateReadingSpec.v
   says why, and has the function that asks the validation of a term
   before so read.

   It commutes with renaming, when what it knows is renamed with the
   term ([load_arrays_knowing_equivariant_statement],
   [load_known_arrays_equivariant_statement]). That it keeps scope is
   in FlambdaArrayLoadScopeSpec.v.

   What is not here: the length of an array that the code does not
   make, known by Simplify from the type of a parameter, of a value
   slot or of what two arms give one continuation; the tests of an
   index under an arm of a [Switch] on the length. Nor does this
   pass make a second load of one element the first: the sharing of
   primitives does (FlambdaCseSpec.shareable has the load of an
   element of an immutable array).

   The proofs are in FlambdaArrayLoadProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNormaliseSpec FlambdaBlockLoadSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* What is known of an array: of each element, what it holds if that is
   known. There is an entry for every element of the array, so the
   number of its elements is known. *)
Definition known_array : Type := list (option simple).

(* The variables and the symbols that are known to be immutable arrays.
   A name may be listed more than once, and then all that is listed of
   it holds. *)
Record known_arrays : Type := Known_arrays {
  known_array_vars : list (variable * known_array);
  known_array_symbols : list (symbol * known_array)
}.

Definition no_arrays : known_arrays := Known_arrays [] [].

(* What is known of an array once [x] is bound again: of the elements
   that held the old [x], nothing. *)
Definition forget_in_array (x : variable) (array : known_array)
    : known_array :=
  map (forget_field x) array.

(* What is known once [x] is bound again: nothing of the variable [x],
   and of the other arrays nothing of the elements that held the old
   [x]. *)
Definition forget_array_var (x : variable) (known : known_arrays)
    : known_arrays :=
  Known_arrays
    (map (fun entry => (fst entry, forget_in_array x (snd entry)))
       (filter (fun entry => negb (Pos.eqb x (fst entry)))
          (known_array_vars known)))
    (map (fun entry => (fst entry, forget_in_array x (snd entry)))
       (known_array_symbols known)).

(* The same of parameters, bound one after the other. *)
Fixpoint forget_array_vars (xs : list variable) (known : known_arrays)
    : known_arrays :=
  match xs with
  | [] => known
  | x :: xs => forget_array_vars xs (forget_array_var x known)
  end.

(* The array that a [Let] makes: the [Make_immutable_array] of the
   model is of an immutable array of values, at any of its array
   kinds. *)
Definition array_made_by (d : named) : option known_array :=
  match d with
  | Prim (Variadic (Make_immutable_array _)) elements =>
    Some (map Some elements)
  | _ => None
  end.

(* What is known in the body of a [Let] of [x] to [d]: nothing that
   mentions the old [x], and that the new one is the array that [d]
   makes, if it makes one. An element of it that is the old [x] is not
   known. *)
Definition arrays_under_let (known : known_arrays) (x : variable)
    (d : named) : known_arrays :=
  let known := forget_array_var x known in
  match array_made_by d with
  | Some array =>
    Known_arrays ((x, forget_in_array x array) :: known_array_vars known)
      (known_array_symbols known)
  | None => known
  end.

(* The array that a static binding binds a symbol to. *)
Definition static_array_made (binding : static_binding)
    : list (symbol * known_array) :=
  match binding with
  | Bind_symbol s (Static_const_immutable_value_array elements) =>
    [(s, map Some elements)]
  | _ => []
  end.

(* What is known in the body of a [Let_static]: its arrays too. *)
Definition arrays_under_statics (known : known_arrays)
    (bindings : list static_binding) : known_arrays :=
  Known_arrays (known_array_vars known)
    (flat_map static_array_made bindings ++ known_array_symbols known).

(* ======================================================================== *)
(* Reading a known array                                                    *)
(* ======================================================================== *)

Definition known_array_of (known : known_arrays) (s : simple)
    : option known_array :=
  match s with
  | Var x => lookup x (known_array_vars known)
  | Symbol sym => lookup sym (known_array_symbols known)
  | Const _ => None
  end.

(* The length of a known array: the number of its elements, where that
   is a tagged immediate (FlambdaPrimitiveSpec.array_length_of). *)
Definition length_of_known (array : known_array) : option simple :=
  let n := Z.of_nat (length array) in
  if in_range (int_width Standard_tagged_immediate) n
  then Some (Const (Tagged_immediate n))
  else None.

(* What a load reads of a known array at an index that is a constant
   tagged immediate: the element at that index, where the array has
   one and what it holds is known. *)
Definition element_loaded (array : known_array) (index : simple)
    : option simple :=
  match index with
  | Const (Tagged_immediate i) =>
    if (0 <=? i)%Z
    then
      match nth_error array (Z.to_nat i) with
      | Some (Some contents) => Some contents
      | Some None | None => None
      end
    else None
  | _ => None
  end.

(* What a primitive computes of a known array, where the term alone
   says it: its length, and a load from it at a constant index. *)
Definition of_known_array (known : known_arrays) (p : prim)
    (args : list simple) : option simple :=
  match p, args with
  | Unary (Array_length _), [array] =>
    match known_array_of known array with
    | Some array => length_of_known array
    | None => None
    end
  | Binary (Array_load _ _ _), [array; index] =>
    match known_array_of known array with
    | Some array => element_loaded array index
    | None => None
    end
  | _, _ => None
  end.

(* A primitive applied to a known array is what it computes of it. *)
Definition load_known_array_named (known : known_arrays) (d : named)
    : named :=
  match d with
  | Prim p args =>
    match of_known_array known p args with
    | Some result => Simple result
    | None => d
    end
  | _ => d
  end.

Fixpoint load_arrays_knowing (known : known_arrays) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (load_known_array_named known d)
      (load_arrays_knowing (arrays_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (load_arrays_knowing (arrays_under_statics known bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (load_arrays_in_handler known h)
      (load_arrays_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (load_arrays_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with load_arrays_in_handler (known : known_arrays)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (load_arrays_knowing (forget_array_vars params known) handler)
  end.

Definition load_known_arrays (e : expr) : expr :=
  load_arrays_knowing no_arrays e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The result refines the term, under all environments. *)
Definition load_known_arrays_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (load_known_arrays e).

(* The pass carries what is known of the arrays of the term, which
   holds variables: those that are known to be arrays, and those that
   the elements of an array are known to hold. A renaming acts on both.
   The symbols that are known to be arrays and the numbers of elements
   are as they were. *)
Definition rename_known_array (r : renaming) (array : known_array)
    : known_array :=
  map (option_map (rename_simple r)) array.

Definition rename_known_arrays (r : renaming) (known : known_arrays)
    : known_arrays :=
  Known_arrays
    (map
       (fun entry =>
          (rename_variable r (fst entry), rename_known_array r (snd entry)))
       (known_array_vars known))
    (map_values (rename_known_array r) (known_array_symbols known)).

(* What is known is renamed with the term. *)
Definition load_arrays_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : known_arrays) (e : expr),
    load_arrays_knowing (rename_known_arrays r known) (rename_expr r e)
    = rename_expr r (load_arrays_knowing known e).

(* The pass with nothing known, as FlambdaValidateReadingSpec.v
   applies it. *)
Definition load_known_arrays_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    load_known_arrays (rename_expr r e)
    = rename_expr r (load_known_arrays e).
