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


(* Loading a field of a block that is known: a normalisation that
   refines.

   This file is to be reviewed: [load_known_blocks] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify replaces the load of a field of a block that it knows by what
   the field holds
   ([Simplify_unary_primitive.simplify_immutable_block_load]), from the
   type of the block; likewise [Is_int] and [Get_tag] of one by their
   answers. Every unit made from an OCaml source ends that way: the
   fields of the module are loaded from the block that the body of the
   module made, to make the static block of the unit. And a function
   inlined where its argument is a constructor applied leaves the match
   on that constructor to be resolved. There are no types of blocks in
   the model yet, so [load_known_blocks] does it for the two cases in
   which a block is known from the term alone. It goes through a term
   keeping ([known_blocks]):

   - for a variable bound by a [Let] to a [Make_block], the tag and the
     arguments of the [Make_block] ([block_made]). The model's
     [Make_block] is the immutable one of values only
     (FlambdaPrimitiveModel.v); a block that is mutable, or of another
     shape, is not converted to it, so nothing is ever known of one;
   - for a symbol bound by a [Let_static] to a [Static_const_block], its
     tag and its fields ([static_block_made]). That too is immutable.

   The two are read alike, through FlambdaValueSpec.block_of, by the
   primitives and by the proof.

   A [Let] of a primitive applied to such a variable or symbol becomes a
   [Let] of a [Simple] ([load_known_named], [of_known_block]):

   - [Is_int], with [variant_only] or without, becomes the naked
     immediate 0;
   - [Get_tag] becomes the tag, a naked immediate;
   - [Block_load kind mut field] becomes what the field holds, under
     these conditions ([field_loaded]), and is left as it is under any
     other: the load is not marked [Mutable]; the access kind says of
     the tag and of the size what is known of the block, where it says
     anything ([known_is]); the block has the field; and the field kind
     of the access is [Any_value]. A load marked [Mutable] is left
     alone. Where Simplify simplifies the primitive it leaves one whose
     argument is not a constant
     ([Simplify_unary_primitive.simplify_mutable_block_load]), and the
     pass, which is for the term before only, does as that does. It
     does not follow the flow analysis of Simplify, which records each
     [Make_block] and each load whose block is a variable
     ([Flow_acc.record_let_binding]) and, for a [Make_block] whose
     variable it requires and does not class as escaping
     ([Mutable_unboxing.blocks_to_unbox]), removes the block and
     replaces each recorded load from it, whatever the load is marked,
     by what the field holds at that point, where the block has the
     field and what it holds is of the load's kind; otherwise it makes
     the load [Invalid]. The pass has no such selection. A load whose
     field kind is [Immediate] is left alone: it has no value on a
     field that is not a tagged immediate, and that is not known from
     the term.

   So where the access kind disagrees with the block the pass does not
   fire, although the load is then undefined
   (FlambdaPrimitiveSpec.eval_block_load) and anything would refine it.

   Nothing else is rewritten. The [Let] stays, as with
   FlambdaAliasesSpec.follow_aliases: the variable is then
   an alias of the field, which that pass follows, and the [Make_block]
   is removed if it is then unused.

   Going under a binder, what is known of its name is forgotten, and so
   is what any field that is that name holds ([forget_var]): the name
   stands for something else from there on. What the other fields of
   such a block hold stays known, and so do its tag and its size: a
   function inlined twice binds the same names twice, and the block made
   by the first copy is read in the second. So nothing is asked of the
   names of the term. A handler of a non-recursive continuation is gone
   into, with what is known where it is defined; the handlers of
   recursive ones and the bodies of code are left as they are
   (FlambdaNormaliseSpec.rewrite_code takes the pass into code, with
   nothing known there).

   The result refines the term ([load_known_blocks_refines_statement]).
   It is not claimed to be equivalent to it: a static block may hold a
   field that is not of kind [Value], the load of which is undefined
   where the field itself is not. So the pass is for the term before
   Simplify.

   The proofs are in FlambdaBlockLoadProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaNormaliseSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* What is known of a block: its tag and, of each field, what it holds if
   that is known. There is an entry for every field of the block, so the
   number of fields is known. *)
Definition known_block : Type := (Z * list (option simple))%type.

(* The variables and the symbols that are known to be blocks. A name may
   be listed more than once, and then all that is listed of it holds. *)
Record known_blocks : Type := Known_blocks {
  known_vars : list (variable * known_block);
  known_symbols : list (symbol * known_block)
}.

Definition no_blocks : known_blocks := Known_blocks [] [].

(* What is known of a field that does not mention [x]. *)
Definition forget_field (x : variable) (field : option simple)
    : option simple :=
  match field with
  | Some s => if var_in_simple x s then None else field
  | None => None
  end.

Definition forget_in_block (x : variable) (block : known_block)
    : known_block :=
  (fst block, map (forget_field x) (snd block)).

(* What is known once [x] is bound again: nothing of the variable [x],
   and of the other blocks nothing of the fields that held the old
   [x]. *)
Definition forget_var (x : variable) (known : known_blocks) : known_blocks :=
  Known_blocks
    (map (fun entry => (fst entry, forget_in_block x (snd entry)))
       (filter (fun entry => negb (Pos.eqb x (fst entry)))
          (known_vars known)))
    (map (fun entry => (fst entry, forget_in_block x (snd entry)))
       (known_symbols known)).

(* The same of parameters, bound one after the other. *)
Fixpoint forget_vars (xs : list variable) (known : known_blocks)
    : known_blocks :=
  match xs with
  | [] => known
  | x :: xs => forget_vars xs (forget_var x known)
  end.

(* The block that a [Let] makes: the model's [Make_block] is of an
   immutable block of values. *)
Definition block_made (d : named) : option known_block :=
  match d with
  | Prim (Variadic (Make_block (Block_kind_values tag))) fields =>
    Some (tag, map Some fields)
  | _ => None
  end.

(* What is known in the body of a [Let] of [x] to [d]: nothing that
   mentions the old [x], and that the new one is the block that [d]
   makes, if it makes one. A field of it that is the old [x] is not
   known. *)
Definition known_under_let (known : known_blocks) (x : variable) (d : named)
    : known_blocks :=
  let known := forget_var x known in
  match block_made d with
  | Some block =>
    Known_blocks ((x, forget_in_block x block) :: known_vars known)
      (known_symbols known)
  | None => known
  end.

(* The block that a static binding binds a symbol to. *)
Definition static_block_made (binding : static_binding)
    : list (symbol * known_block) :=
  match binding with
  | Bind_symbol s (Static_const_block tag fields) =>
    [(s, (tag, map Some fields))]
  | _ => []
  end.

(* What is known in the body of a [Let_static]: its blocks too. *)
Definition known_under_statics (known : known_blocks)
    (bindings : list static_binding) : known_blocks :=
  Known_blocks (known_vars known)
    (flat_map static_block_made bindings ++ known_symbols known).

(* ======================================================================== *)
(* Loading from a known block                                               *)
(* ======================================================================== *)

Definition known_block_of (known : known_blocks) (s : simple)
    : option known_block :=
  match s with
  | Var x => lookup x (known_vars known)
  | Symbol sym => lookup sym (known_symbols known)
  | Const _ => None
  end.

(* What a load reads of a known block, where the access kind agrees with
   what is known of the block, the block has the field, what the field
   holds is known, and the access asks nothing of it. *)
Definition field_loaded (kind : block_access_kind) (field : Z)
    (block : known_block) : option simple :=
  match kind with
  | Block_access_values tag size Any_value =>
    if known_is tag (fst block)
       && known_is size (Z.of_nat (length (snd block)))
       && (0 <=? field)%Z
    then
      match nth_error (snd block) (Z.to_nat field) with
      | Some (Some contents) => Some contents
      | Some None | None => None
      end
    else None
  | Block_access_values _ _ Immediate => None
  end.

(* What a unary primitive computes of a known block, where the term
   alone says it: a load, the field; [Is_int], no; [Get_tag], the tag. *)
Definition of_known_block (p : unary_primitive) (block : known_block)
    : option simple :=
  match p with
  | Block_load kind (Immutable | Immutable_unique) field =>
    field_loaded kind field block
  | Is_int _ => Some (Const (Naked_immediate 0))
  | Get_tag => Some (Const (Naked_immediate (fst block)))
  | _ => None
  end.

(* A primitive applied to arguments, as a unary primitive applied to
   one. *)
Definition unary_applied (p : prim) (args : list simple)
    : option (unary_primitive * simple) :=
  match p, args with
  | Unary p, [arg] => Some (p, arg)
  | _, _ => None
  end.

(* A primitive applied to a known block is what it computes of it. *)
Definition load_known_named (known : known_blocks) (d : named) : named :=
  match d with
  | Prim p args =>
    match unary_applied p args with
    | Some (p, block) =>
      match known_block_of known block with
      | Some known_block =>
        match of_known_block p known_block with
        | Some result => Simple result
        | None => d
        end
      | None => d
      end
    | None => d
    end
  | _ => d
  end.

Fixpoint load_blocks_knowing (known : known_blocks) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (load_known_named known d)
      (load_blocks_knowing (known_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (load_blocks_knowing (known_under_statics known bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (load_blocks_in_handler known h)
      (load_blocks_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (load_blocks_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with load_blocks_in_handler (known : known_blocks) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (load_blocks_knowing (forget_vars params known) handler)
  end.

Definition load_known_blocks (e : expr) : expr :=
  load_blocks_knowing no_blocks e.

Definition load_known_blocks_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (load_known_blocks e).
