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


(* Unboxing a number that is known to be boxed, and untagging one that
   is known to be tagged: a normalisation that refines.

   This file is to be reviewed: [unbox_known_numbers] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify replaces the unboxing of a boxed number that it knows by the
   number ([Simplify_unary_primitive.simplify_unbox_number]), and the
   untagging of a tagged immediate that it knows by the naked immediate
   ([simplify_untag_immediate]), from the type of the argument.
   Arithmetic on boxed numbers is made of the first: each operation
   unboxes its arguments and boxes its result, and the result of one is
   the argument of the next. Every conditional is made of the second:
   the front end tags the result of a comparison, a boolean, and untags
   it for the [Switch]. This normalisation uses no types: it does the
   same where the number is known from the term alone, by a fact that
   can be read off a [Let] or a static binding. It goes through a term
   keeping ([known_numbers]):

   - for a variable bound by a [Let] to [Box_number kind] or to
     [Tag_immediate] applied to a [Simple], how it was wrapped and that
     [Simple] ([number_wrapped]);
   - for a symbol bound by a [Let_static] to a [Static_const_boxed_number]
     of a constant, the constant ([static_number_boxed]). A static boxed
     number whose contents are a variable is not kept.

   A [Let] of [Unbox_number kind] or of [Untag_immediate] applied to
   such a variable or symbol becomes a [Let] of the contents
   ([unbox_known_named], [unwrapped]), where the unwrapping is the
   converse of the wrapping (an unboxing at the kind of the boxing, an
   untagging of a tagging), or the constant is a number of the kind of
   the unboxing (FlambdaPrimitiveSpec.is_number_of_kind), and is left as
   it is otherwise. Nothing else is rewritten, and the [Let] stays, as
   in FlambdaBlockLoadSpec.v, which this file follows: what is known of
   a name, and what mentions it, is forgotten under a binder of it; a
   handler of a non-recursive continuation is gone into, with what is
   known where it is defined; the handlers of recursive ones and the
   bodies of code are left as they are.

   The result refines the term
   ([unbox_known_numbers_refines_statement]). It is not claimed to be
   equivalent to it, though it most likely is; a refinement is what the
   term before Simplify needs.

   The proofs are in FlambdaUnboxProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaBlockLoadSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* How a number is wrapped: boxed at a kind, or tagged. *)
Inductive wrapping : Type :=
  | Boxed (kind : boxable_number)
  | Tagged.

Definition wrapping_eq_dec : ∀ w w' : wrapping, {w = w'} + {w <> w'}.
Proof. repeat decide equality. Defined.

Definition wrapping_eqb (w w' : wrapping) : bool :=
  if wrapping_eq_dec w w' then true else false.

(* The variables that are known to be wrapped numbers, each with how it
   was wrapped and what was wrapped; and the symbols that are boxed
   numbers, each with the constant in the box. *)
Record known_numbers : Type := Known_numbers {
  known_wrapped_vars : list (variable * (wrapping * simple));
  known_boxed_symbols : list (symbol * const)
}.

Definition no_numbers : known_numbers := Known_numbers [] [].

(* What is known that does not mention [x]. Nothing known of a symbol
   mentions a variable. *)
Definition forget_wrapped_var (x : variable) (known : known_numbers)
    : known_numbers :=
  Known_numbers
    (filter
       (fun entry =>
          negb (Pos.eqb x (fst entry))
          && negb (var_in_simple x (snd (snd entry))))
       (known_wrapped_vars known))
    (known_boxed_symbols known).

(* The same of parameters, bound one after the other. *)
Fixpoint forget_wrapped_vars (xs : list variable) (known : known_numbers)
    : known_numbers :=
  match xs with
  | [] => known
  | x :: xs => forget_wrapped_vars xs (forget_wrapped_var x known)
  end.

(* The number that a [Let] boxes or tags, with how. *)
Definition number_wrapped (d : named) : option (wrapping * simple) :=
  match d with
  | Prim p args =>
    match unary_applied p args with
    | Some (Box_number kind, contents) => Some (Boxed kind, contents)
    | Some (Tag_immediate, contents) => Some (Tagged, contents)
    | _ => None
    end
  | _ => None
  end.

(* What is known in the body of a [Let] of [x] to [d]: nothing that
   mentions the old [x], and that the new one is the number that [d]
   wraps, if it wraps one that is not the old [x]. *)
Definition numbers_under_let (known : known_numbers) (x : variable)
    (d : named) : known_numbers :=
  let known := forget_wrapped_var x known in
  match number_wrapped d with
  | Some (how, contents) =>
    if var_in_simple x contents then known
    else
      Known_numbers ((x, (how, contents)) :: known_wrapped_vars known)
        (known_boxed_symbols known)
  | None => known
  end.

(* The constant that a static binding boxes. *)
Definition static_number_boxed (binding : static_binding)
    : list (symbol * const) :=
  match binding with
  | Bind_symbol s (Static_const_boxed_number (Const c)) => [(s, c)]
  | _ => []
  end.

(* What is known in the body of a [Let_static]: its boxed constants
   too. *)
Definition numbers_under_statics (known : known_numbers)
    (bindings : list static_binding) : known_numbers :=
  Known_numbers (known_wrapped_vars known)
    (flat_map static_number_boxed bindings ++ known_boxed_symbols known).

(* ======================================================================== *)
(* Unwrapping a known number                                                *)
(* ======================================================================== *)

(* What is in a known wrapped number, where the unwrapping is the
   converse of the wrapping. *)
Definition unwrapped (known : known_numbers) (how : wrapping)
    (wrapped : simple) : option simple :=
  match wrapped with
  | Var x =>
    match lookup x (known_wrapped_vars known) with
    | Some (how', contents) =>
      if wrapping_eqb how how' then Some contents else None
    | None => None
    end
  | Symbol s =>
    match how, lookup s (known_boxed_symbols known) with
    | Boxed kind, Some c =>
      if is_number_of_kind kind c then Some (Const c) else None
    | _, _ => None
    end
  | Const _ => None
  end.

(* The number unwrapped from a known wrapped number, of a primitive
   applied to arguments that is such an unwrapping. *)
Definition unwrapping_of_known (known : known_numbers) (p : prim)
    (args : list simple) : option simple :=
  match unary_applied p args with
  | Some (Unbox_number kind, boxed) => unwrapped known (Boxed kind) boxed
  | Some (Untag_immediate, tagged) => unwrapped known Tagged tagged
  | _ => None
  end.

Definition unbox_known_named (known : known_numbers) (d : named) : named :=
  match d with
  | Prim p args =>
    match unwrapping_of_known known p args with
    | Some contents => Simple contents
    | None => d
    end
  | _ => d
  end.

Fixpoint unbox_numbers_knowing (known : known_numbers) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (unbox_known_named known d)
      (unbox_numbers_knowing (numbers_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (unbox_numbers_knowing (numbers_under_statics known bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (unbox_numbers_in_handler known h)
      (unbox_numbers_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (unbox_numbers_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with unbox_numbers_in_handler (known : known_numbers)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (unbox_numbers_knowing (forget_wrapped_vars params known) handler)
  end.

Definition unbox_known_numbers (e : expr) : expr :=
  unbox_numbers_knowing no_numbers e.

Definition unbox_known_numbers_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (unbox_known_numbers e).
