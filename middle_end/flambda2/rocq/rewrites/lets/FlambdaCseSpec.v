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

(* Not computing again a primitive that has been computed: a
   normalisation that refines.

   This file is to be reviewed: [share_known_primitives] and what it is
   made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; what it means for what the
   pass knows to hold ([all_computed], [all_computed_or_made]); the
   action of a renaming on what the pass knows ([rename_computed]);
   and the statements.

   Simplify keeps the primitives it has seen applied, each with the
   variable bound to its result, and replaces a later application of the
   same primitive to the same arguments by that variable (common
   subexpression elimination: [Simplify_primitive], with
   [Flambda_primitive.Eligible_for_cse]). A boxed number that is used
   twice is unboxed twice by the front end and once after Simplify; so is
   a comparison or a sum written twice in the source, once a function
   has been inlined.

   [share_known_primitives] does the same with what can be read off the
   term. It goes through a term keeping, for a variable bound by a [Let]
   to a primitive that may be shared ([shareable]) applied to arguments,
   that primitive and those arguments ([computed]). A later [Let] of the
   same primitive applied to the same arguments becomes a [Let] of that
   variable ([share_named]). Nothing else is rewritten, and the [Let]
   stays, as in FlambdaBlockLoadSpec.v and FlambdaUnboxSpec.v, which this
   file follows: what is known of a variable, and what mentions it in an
   argument, is forgotten under a binder of it; a handler of a
   non-recursive continuation is gone into, with what is known where it
   is defined; the handlers of recursive ones and the bodies of code are
   left as they are.

   One thing more is known under a [Let] of the untagging of a
   variable: that the variable holds the tagging of what the [Let]
   binds ([tagged_again]). A later [Let] of that tagging then becomes
   a [Let] of the variable. Simplify records the same equation where
   it simplifies an untagging
   ([Simplify_unary_primitive.simplify_untag_immediate], the inverse
   equation for its sharing). The front end untags a variant that is
   not a block to test it; where the arms give each constant
   constructor its own number they pass tagged constants, the merging
   of those arms makes the tagging of what was tested
   (FlambdaMergeSwitchesSpec.merge_switches), and this equation makes
   that tagging the variable. The like for the boxing of
   an unboxed number is not known here: Simplify records it only
   where it has proved that the box is on the heap, and in the model
   a boxed number that is a symbol and one that is made are not the
   same value.

   Another thing is known under a [Let] of a variable to
   [Make_mutable_array], the allocation of a mutable array: that the
   variable holds the array which that allocation made of its
   arguments ([array_made_under_let]). Of this only the number of the
   arguments is read. A later [Let] of the length of that variable,
   at any length kind, becomes a [Let] of that number, where the
   number is a tagged immediate ([made_length_named]): the elements
   of such an array may be written, and its length does not change
   (FlambdaHeapSpec.array_length_known_statement). Simplify has the
   same: [Simplify_variadic_primitive.simplify_make_array]
   gives the variable the type of a mutable array of that length, and
   [Simplify_unary_primitive.simplify_array_length] reads the length
   in the type. Nothing is known of an array that no [Let] in scope
   makes: of a parameter, of what a call returns, of what is read
   from a block or from a closure. FlambdaMadeLengthsSpec.v has that
   rule alone, for the rounds of the term after Simplify.

   What may be shared is a choice that is narrower than the compiler's.
   The primitives listed in [shareable] compute a number from numbers,
   or look at a value without reading memory that can change: the
   length of a string is among them, as the compiler shares it too
   ([String_length] has no effects and no coeffects in
   [Flambda_primitive]); so is the length of an array, which the
   compiler shares ([Flambda_primitive.unary_primitive_eligible_for_cse]
   says yes of [Array_length]) and which does not change once the
   array is made, whether the array can be written or not (here the
   primitive has a value on an immutable array, on a symbol bound to
   one and on an array that is a cell of the heap; of such a cell,
   FlambdaHeapSpec.heap_evolves says that it keeps its number of
   elements); and so is a [Block_load] marked [Immutable] or
   [Immutable_unique], a load of a field that is not written
   (FlambdaPrimitiveSpec.v says what the mark is for). The compiler
   has the second of two such loads stand for the first by another
   road: [Flambda_primitive.unary_primitive_eligible_for_cse] says no
   of every [Block_load], and a load from an immutable block leaves
   in the typing environment what the type of the block and the shape
   that names the variable of the load have in common
   ([Simplify_common.simplify_projection]), where a later load of the
   field finds that variable. And so is an [Array_load] marked
   [Immutable] or [Immutable_unique], the load of an element of an
   immutable array, which the compiler's sharing takes
   ([Flambda_primitive.binary_primitive_eligible_for_cse] says yes
   of it), so that Simplify may make a second load of one element
   of one array the variable of the first. Such a load has a value
   on an immutable array and on a symbol bound to one, which nothing
   writes. It has no value on an array that has an identity, as a
   [Block_load] so marked has none on a block made [Mutable]: the
   front end makes one for the reads of an immutable array, and
   such an array may be a mutable array seen as immutable
   ([%array_to_iarray], which is the identity); the model refuses
   that program (FlambdaHeapSpec.exec_binary_on_cell, with its note;
   decision D49). The sharing rests on that decision as it stands:
   were such a load to read a cell that a later store may change,
   in a run that has a meaning, the two loads could differ and it
   would leave the list. Left out: [Opaque_identity], which Simplify
   never shares; a [Block_load] and an [Array_load] marked
   [Mutable], which are not shared; [String_or_bigstring_load],
   until the model says which strings cannot change; and the
   primitives that allocate a block or an array ([Make_block],
   [Make_immutable_array]), which the compiler does share where
   what is allocated is immutable.

   The allocation of a mutable array is not shared either, here or
   in the compiler: [share_named] looks up only a primitive that may
   be shared. Of a mutable array that a [Let] makes the length alone
   is known (above), and not its elements, which a store may change.

   [Box_number] allocates too, and is in the list: a later boxing of
   the same naked number becomes the variable of the first, as in the
   compiler ([Box_number] is eligible for its sharing). A function
   that gives its boxed operand back in two results boxes it twice
   before Simplify and once after. In the model a boxed number is its
   contents and no more (FlambdaValueSpec.v): two boxes of one number
   are one value, and [Phys_equal] has no value on two boxed numbers
   (FlambdaPrimitiveSpec.eval_phys_equal: neither has an identity).
   So no run that has a meaning tells one box from two, and the
   statements below say of the pass with [Box_number] in the list
   what they said without it.

   CR mshinwell: sharing an allocation makes two values one, which
   physical equality can tell apart in the program. The compiler
   shares immutable allocations all the same. This file follows it
   for boxed numbers only: that is sound in the model because
   [Phys_equal] has no value on two allocated values (decision D19),
   so the model says nothing of a run that executes a comparison of
   those two boxed values. Blocks and arrays are not shared here. The
   other answer would be that the validation never makes two
   allocations one: [Box_number] would leave [shareable], and a pair
   of terms whose validation needs this sharing could then stay not
   validated. Decision D42.

   The result refines the term
   ([share_known_primitives_refines_statement]). That it is equivalent
   to it is
   FlambdaCseEquivalentSpec.share_known_primitives_equivalent_statement;
   a refinement is what the term before Simplify needs.

   The pass commutes with the renamings of FlambdaNominalSpec.v, what it
   knows being renamed with the term
   ([share_primitives_knowing_equivariant_statement]): it compares
   variables for equality and makes no name.

   The proofs are in FlambdaCseProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* The primitives whose result may stand for a later application to the
   same arguments. *)
(* The last case, the load of an element of an immutable array, rests
   on decision D49 as it stands: an [Array_load] marked [Immutable]
   or [Immutable_unique] has no value on an array that has an
   identity (FlambdaHeapSpec.exec_binary_on_cell), so where the first
   of two such loads has a value its array is an immutable array or a
   symbol bound to one, and the second has the same. If that decision
   comes to give such a load a value on a cell that a later store may
   change, in a run that has a meaning, the case goes out. *)
Definition shareable (p : prim) : bool :=
  match p with
  | Unary
      (Is_int _ | Get_tag | Num_conv _ _ | Boolean_not | Unbox_number _
      | Box_number _
      | Untag_immediate | Tag_immediate | String_length | Array_length _
      | Unary_int_arith _ _
      | Block_load _ (Immutable | Immutable_unique) _) =>
    true
  | Binary (Phys_equal _ | Int_arith _ _ | Int_shift _ _ | Int_comp _ _) =>
    true
  | Binary (Array_load _ _ (Immutable | Immutable_unique)) => true
  | _ => false
  end.

(* The variables known to hold the result of a primitive, each with the
   primitive and its arguments.

   In this pass the list has entries of two sorts: of a primitive that
   may be shared, with a variable that holds its result
   ([computed_under_let]); and of the allocation of a mutable array,
   with the variable that holds the array ([array_made_under_let]), of which
   only the number of the arguments is read. Besides this pass,
   FlambdaMadeLengthsSpec.lengths_knowing builds such entries, with
   [array_made_under_let], and none of the other sort. *)
Definition computed : Type := list (variable * (prim * list simple)).

(* What is known that does not mention [x]. An entry of the allocation
   of a mutable array goes as the others do, when its variable or one
   of its arguments is bound again: for an argument that is more than
   the length of the array needs, and forgetting an entry is always
   sound. *)
Definition forget_computed (x : variable) (known : computed) : computed :=
  filter
    (fun entry =>
       negb (Pos.eqb x (fst entry))
       && negb (var_in_simples x (snd (snd entry))))
    known.

(* The same of parameters, bound one after the other. *)
Fixpoint forget_computed_vars (xs : list variable) (known : computed)
    : computed :=
  match xs with
  | [] => known
  | x :: xs => forget_computed_vars xs (forget_computed x known)
  end.

(* What the untagging of a variable says of that variable: where [x]
   is bound to [p] applied to [args] and that is the untagging of the
   variable [t], [t] holds the tagging of [x]. Nothing of any other
   primitive. *)
Definition tagged_again (x : variable) (p : prim) (args : list simple)
    : computed :=
  match p, args with
  | Unary Untag_immediate, [Var t] => [(t, (Unary Tag_immediate, [Var x]))]
  | _, _ => []
  end.

(* What is known in the body of a [Let] of [x] to [d]: nothing that
   mentions the old [x], and that the new one is the result of [d], if
   that is a primitive that may be shared, applied to arguments that do
   not mention the old [x]; and then what [tagged_again] adds. *)
Definition computed_under_let (known : computed) (x : variable) (d : named)
    : computed :=
  let known := forget_computed x known in
  match d with
  | Prim p args =>
    if shareable p && negb (var_in_simples x args)
    then (x, (p, args)) :: tagged_again x p args ++ known
    else known
  | _ => known
  end.

(* What it means for what is known to hold where a term runs, at a
   heap and a read record: the primitive is one whose value stays
   (FlambdaHeapSpec.stable_prim); and run now on its arguments, as the
   machine runs it at a [Let] (FlambdaSemanticsSpec.exec_named), it
   gives the value that the variable is bound to, and leaves the heap
   and the record as they are.

   The heap and the record are those that the [Let] by which it is
   known left, or any that they have evolved to since
   (FlambdaSemanticsSpec.memory_evolves): the fact is of a primitive
   that has run, and says that to run it again changes neither. *)
Definition computed_holds (statics : static_env) (hp : heap) (rd : read_record)
    (vars : env) (entry : variable * (prim * list simple)) : Prop :=
  stable_prim (fst (snd entry)) = true
  ∧ ∃ v : value,
      exec_named statics hp rd vars (Prim (fst (snd entry)) (snd (snd entry)))
      = Some (v, hp, rd)
      ∧ lookup (fst entry) vars = Some v.

Definition all_computed (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) : Prop :=
  ∀ entry : variable * (prim * list simple),
    entry ∈ known -> computed_holds statics hp rd vars entry.

(* Every primitive that may be shared is one whose value stays: what
   [computed_under_let] adds is of such a primitive, as [computed_holds]
   asks. *)
Definition shareable_stable_statement : Prop :=
  ∀ p : prim, shareable p = true -> stable_prim p = true.

(* ======================================================================== *)
(* The arrays that are known to have been made                              *)
(* ======================================================================== *)

(* What a [Let] of [x] to the allocation of a mutable array makes
   known: that [x] holds the array which the allocation made of these
   arguments. As for a primitive that may be shared
   ([computed_under_let]), nothing where an argument mentions the old
   [x]. The entry is kept with the others and forgotten as they are
   ([forget_computed]). Its primitive is not one that may be shared,
   and of such an entry only the number of its arguments is read
   ([made_length]). *)
Definition array_made_under_let (x : variable) (d : named) : computed :=
  match d with
  | Prim (Heap (Make_mutable_array kind)) args =>
    if var_in_simples x args
    then []
    else [(x, (Heap (Make_mutable_array kind), args))]
  | _ => []
  end.

(* What the sharing knows in the body of a [Let] of [x] to [d]: what
   [computed_under_let] says, and the array that [d] makes. *)
Definition computed_or_made_under_let (known : computed) (x : variable)
    (d : named) : computed :=
  array_made_under_let x d ++ computed_under_let known x d.

(* What it means for an entry of the allocation of a mutable array to
   hold where a term runs, at a heap: the variable is bound to what
   the allocation gave when it ran, on as many values as the entry has
   arguments; and the heap is the one that the allocation left, or one
   that it has evolved to since (FlambdaHeapSpec.heap_evolves). The
   length of that array is then that number, at any length kind and in
   that heap, where the number is a tagged immediate
   (FlambdaHeapSpec.array_length_known_statement). Nothing is said of
   the elements. The read record is not in it: it says what an
   allocation gave and what the heap is, and FlambdaHeapSpec.exec_prim,
   in whose terms both are said, has no record. *)
Definition array_made_holds (statics : static_env) (hp : heap) (vars : env)
    (entry : variable * (prim * list simple)) : Prop :=
  ∃ (kind : array_kind) (h h1 : heap) (vs : list value) (v : value),
    fst (snd entry) = Heap (Make_mutable_array kind)
    ∧ length vs = length (snd (snd entry))
    ∧ exec_prim statics h (Heap (Make_mutable_array kind)) vs = Some (v, h1)
    ∧ heap_evolves h1 hp
    ∧ lookup (fst entry) vars = Some v.

(* What the sharing knows holds: each entry in one of the two ways. An
   entry of a primitive that may be shared can hold in the first way
   only, and an entry of the allocation of a mutable array in the
   second only: that allocation is not a primitive whose value stays. *)
Definition all_computed_or_made (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) : Prop :=
  ∀ entry : variable * (prim * list simple),
    entry ∈ known ->
    computed_holds statics hp rd vars entry
    ∨ array_made_holds statics hp vars entry.

(* ======================================================================== *)
(* Sharing                                                                  *)
(* ======================================================================== *)

Definition same_simples (args args' : list simple) : bool :=
  if list_eq_dec simple_eq_dec args args' then true else false.

(* The first variable known to hold the result of [p] applied to
   [args]. *)
Fixpoint computed_by (known : computed) (p : prim) (args : list simple)
    : option variable :=
  match known with
  | [] => None
  | (x, (p', args')) :: known =>
    if prim_eqb p p' && same_simples args args'
    then Some x
    else computed_by known p args
  end.

(* The number of the arguments of the first entry that says that [x]
   holds a mutable array made of them. A list that the pass builds
   has one such entry at most for a variable: what is known of the
   old variable is forgotten before the new entry is added. *)
Fixpoint made_length (known : computed) (x : variable) : option Z :=
  match known with
  | [] => None
  | (y, (Heap (Make_mutable_array _), args)) :: known =>
    if Pos.eqb x y
    then Some (Z.of_nat (length args))
    else made_length known x
  | _ :: known => made_length known x
  end.

(* The length, at any length kind, of a variable that is known to hold
   an array made of so many arguments: that number as a constant,
   where it is a tagged immediate, which is where
   FlambdaPrimitiveSpec.array_length_of has an answer of so many
   elements, and that is its answer. Nothing of any other defining
   expression. *)
Definition made_length_named (known : computed) (d : named) : option named :=
  match d with
  | Prim (Unary (Array_length _)) [Var x] =>
    match made_length known x with
    | Some n =>
      if in_range (int_width Standard_tagged_immediate) n
      then Some (Simple (Const (Tagged_immediate n)))
      else None
    | None => None
    end
  | _ => None
  end.

(* The length of an array that is known to have been made is its
   constant. Otherwise a primitive that may be shared is looked up
   among what is known, and a primitive that may not be shared is not
   looked up.

   That test is what keeps two mutable arrays from being made one:
   without it a [Let] of [Make_mutable_array] applied to the arguments
   of an entry of such an allocation would become the variable of that
   entry. (The boxing of a number is an allocation that may be shared:
   the header says why.) With it, a [Let] of a primitive that may not
   be shared is never rewritten, whatever entries [known] has; the
   statements of renaming are of every [known]. From nothing known,
   as the validation runs the pass, the test changes nothing on a
   term that has no [Let] of a mutable array: every entry is then one
   that [computed_under_let] added, of a primitive that may be
   shared, so a lookup that answers is of such a primitive. *)
Definition share_named (known : computed) (d : named) : named :=
  match made_length_named known d with
  | Some d' => d'
  | None =>
    match d with
    | Prim p args =>
      if shareable p
      then
        match computed_by known p args with
        | Some x => Simple (Var x)
        | None => d
        end
      else d
    | _ => d
    end
  end.

Fixpoint share_primitives_knowing (known : computed) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (share_named known d)
      (share_primitives_knowing (computed_or_made_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings (share_primitives_knowing known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (share_primitives_in_handler known h)
      (share_primitives_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (share_primitives_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with share_primitives_in_handler (known : computed)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (share_primitives_knowing (forget_computed_vars params known) handler)
  end.

Definition share_known_primitives (e : expr) : expr :=
  share_primitives_knowing [] e.

Definition share_known_primitives_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (share_known_primitives e).

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* What is known holds variables: the one bound to each result, and
   those among the arguments. A renaming acts on both. The primitive has
   no name in it. *)
Definition rename_computed (r : renaming) (known : computed) : computed :=
  map
    (fun entry =>
       (rename_variable r (fst entry),
        (fst (snd entry), map (rename_simple r) (snd (snd entry)))))
    known.

(* What is known is renamed with the term. *)
Definition share_primitives_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : computed) (e : expr),
    share_primitives_knowing (rename_computed r known) (rename_expr r e)
    = rename_expr r (share_primitives_knowing known e).

(* The pass as the validation runs it, with nothing known. *)
Definition share_known_primitives_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    share_known_primitives (rename_expr r e)
    = rename_expr r (share_known_primitives e).
