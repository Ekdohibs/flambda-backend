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

(* Reading a known block where the read is sure to have a value, and
   removing an allocation of constants that nothing uses: two
   normalisations that are equivalences.

   This file is to be reviewed: [surely_value], [surely_immediate],
   [load_known_blocks_surely], [remove_constant_allocations] and what
   they are made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; and the statements.

   The term before Simplify is normalised by refinements, among them
   FlambdaBlockLoadSpec.load_known_blocks, which puts what a field of a
   known block holds in place of its load, and
   FlambdaNormaliseSpec.remove_unused_lets, which then removes the
   [Let] of a block that nothing reads any more. The term after may be
   rewritten by equivalences only. Simplify leaves a load from a block
   that it does not know where it stands: one behind a call that it
   did not inline. Where the validation inlines that call in both
   terms, the term before loses the load and the block, and the term
   after keeps both.

   Why the passes of the term before are not equivalences.
   - A load has no value on a field that is not of the kind it reads
     (FlambdaPrimitiveSpec.eval_block_load): a load of a field at the
     field kind [Any_value] wants a value of kind [Value]. A static
     block holds whatever its binding says, so the term may be
     undefined where the field itself is not.
   - A [Let] has no step where its defining expression has no value
     (FlambdaSemanticsSpec.step). [Make_block] and
     [Make_immutable_array] have none on an argument that is not of
     kind [Value], on a symbol that is bound to nothing, and, for an
     array of immediates, on what is not a tagged immediate in range
     (FlambdaPrimitiveSpec.eval_variadic_primitive). Removing such a
     [Let] would give a meaning to a term that had none.

   What is here, each with a test on the term that makes it an
   equivalence. What is known of the blocks of a term is kept as the
   pass of the term before keeps it
   (FlambdaBlockLoadSpec.known_blocks): for a variable bound by a
   [Let] that has taken its step, and for a symbol bound by a
   [Let_static] around.
   - [load_known_blocks_surely], the part of
     FlambdaBlockLoadSpec.load_known_blocks that is sure to have a
     value. [Is_int] and [Get_tag] of a known block are read as
     FlambdaBlockLoadSpec.of_known_block reads them: each has a
     value on every block. A load is read under the conditions of
     FlambdaBlockLoadSpec.field_loaded, and one more: what the field is
     known to hold is of kind [Value] by its form, a symbol or a
     constant tagged immediate ([surely_value]). The load then has a
     value, the one of the field. A field known to hold a variable is
     not read: the variable is in scope and has a value, but nothing in
     the term says of what kind.
   - [remove_constant_allocations]. A [Let] at the kind [Value] of a
     [Make_block] or a [Make_immutable_array] every argument of which
     is a constant tagged immediate in range, whose variable the body
     does not use, is removed. Such an allocation has a value whatever
     the variables and the symbols are bound to, and it is of kind
     [Value]. An allocation with a variable or a symbol among its
     arguments is left: a variable may hold a value of another kind,
     and a symbol may be bound to nothing.
     FlambdaNormaliseSpec.named_defined, which says of no primitive
     that it has a value, is not changed.

   The read leaves the [Let]: the variable is then an alias of what
   was read, which FlambdaAliasesSpec.follow_aliases
   follows, and the allocation is removed by the second pass if
   nothing else uses it.

   As the pass of the term before: under a binder what is known of its
   name is forgotten; a handler of a non-recursive continuation is
   gone into, with what is known where it is defined; the handlers of
   recursive ones and the bodies of code are left as they are.

   Not done.
   - A field known to hold a variable, of a block that a [Let] made.
     Such a block holds values of kind [Value] only, so the read would
     be an equivalence. Its proof needs, of a variable known to be a
     block, that every field of the block is of kind [Value]; what the
     proof of the pass of the term before keeps of such a variable
     does not say it.
   - The length of a known immutable array and a load from one. As
     equivalences (the length always; a load under a test on the
     index and on the element) they were proved and tried in the
     rounds of the term after, and are not in this tree: with them
     the validation refused units that it accepts. It reads the
     arrays of the term before beside its rounds and not in them
     (FlambdaValidateReadingSpec.v), so in the rounds the term after
     had the constant where the term before has the primitive.

   The results are equivalent to the term
   ([load_known_blocks_surely_equivalent_statement],
   [remove_constant_allocations_equivalent_statement]).

   The proofs are in FlambdaReadsAfterProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaNominalSpec FlambdaBlockLoadSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a simple is, by its form                                            *)
(* ======================================================================== *)

(* A simple whose value, if it has one, is of kind [Value]: a symbol,
   bound or not, and a constant tagged immediate, in range or not. *)
Definition surely_value (s : simple) : bool :=
  match s with
  | Symbol _ | Const (Tagged_immediate _) => true
  | Var _ | Const _ => false
  end.

(* A constant tagged immediate that is in range: its value is a tagged
   immediate (FlambdaPrimitiveSpec.is_tagged_immediate). *)
Definition surely_immediate (s : simple) : bool :=
  match s with
  | Const (Tagged_immediate i) =>
    in_range (int_width Standard_tagged_immediate) i
  | Var _ | Symbol _ | Const _ => false
  end.

(* ======================================================================== *)
(* Reading a known block                                                    *)
(* ======================================================================== *)

(* What a unary primitive computes of a known block, where it is sure
   to have a value: [Is_int] and [Get_tag] as
   FlambdaBlockLoadSpec.of_known_block has them, and a load where what
   the field holds is of kind [Value] by its form. *)
Definition of_known_block_surely (p : unary_primitive)
    (block : known_block) : option simple :=
  match p with
  | Block_load _ _ _ =>
    match of_known_block p block with
    | Some contents => if surely_value contents then Some contents else None
    | None => None
    end
  | Is_int _ | Get_tag => of_known_block p block
  | _ => None
  end.

Definition load_known_named_surely (known : known_blocks) (d : named) : named :=
  match d with
  | Prim p args =>
    match unary_applied p args with
    | Some (p, block) =>
      match known_block_of known block with
      | Some known_block =>
        match of_known_block_surely p known_block with
        | Some result => Simple result
        | None => d
        end
      | None => d
      end
    | None => d
    end
  | _ => d
  end.

Fixpoint load_blocks_knowing_surely (known : known_blocks) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (load_known_named_surely known d)
      (load_blocks_knowing_surely (known_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (load_blocks_knowing_surely (known_under_statics known bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (load_blocks_in_handler_surely known h)
      (load_blocks_knowing_surely known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (load_blocks_knowing_surely known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with load_blocks_in_handler_surely (known : known_blocks)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (load_blocks_knowing_surely (forget_vars params known) handler)
  end.

Definition load_known_blocks_surely (e : expr) : expr :=
  load_blocks_knowing_surely no_blocks e.

Definition load_known_blocks_surely_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (load_known_blocks_surely e).

(* Where it reads, it reads what the pass of the term before reads. *)
Definition load_known_named_surely_loads_statement : Prop :=
  ∀ (known : known_blocks) (d : named),
    load_known_named_surely known d = d
    ∨ load_known_named_surely known d = load_known_named known d.

(* ======================================================================== *)
(* Removing an allocation of constants that nothing uses                    *)
(* ======================================================================== *)

(* A [Make_block] or a [Make_immutable_array] every argument of which
   is a constant tagged immediate in range. *)
Definition allocation_of_constants (d : named) : bool :=
  match d with
  | Prim (Variadic (Make_block (Block_kind_values _))) args
  | Prim (Variadic (Make_immutable_array _)) args =>
    forallb surely_immediate args
  | _ => false
  end.

Fixpoint remove_constant_allocations (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let body' := remove_constant_allocations body in
    if kind_eqb kind Value
       && allocation_of_constants d
       && negb (var_free_b x body')
    then body'
    else Let x kind d body'
  | Let_static bindings body =>
    Let_static bindings (remove_constant_allocations body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (remove_constant_allocations handler))
      (remove_constant_allocations body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (remove_constant_allocations body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition remove_constant_allocations_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (remove_constant_allocations e).

(* Such an allocation has a value of kind [Value], whatever the
   variables and the symbols are bound to. *)
Definition allocation_of_constants_defined_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (d : named),
    allocation_of_constants d = true ->
    ∃ v : value,
      eval_named statics vars d = Some v ∧ kind_of_value v = Value.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The two passes compare variables for equality and make no name, so
   each commutes with the renamings of FlambdaNominalSpec.v. The
   statement for the read of a known block is with the action of a
   renaming on what is known of the blocks, in
   FlambdaValidatorEquivarianceSpec.v. *)
Definition remove_constant_allocations_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    remove_constant_allocations (rename_expr r e)
    = rename_expr r (remove_constant_allocations e).
