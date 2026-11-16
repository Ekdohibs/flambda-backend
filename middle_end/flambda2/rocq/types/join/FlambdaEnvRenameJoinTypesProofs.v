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



(* The traversal of types of the join on renamed arguments. Nothing
   here needs review: it holds lemmas only, and no statement.

   For each function of FlambdaJoinTypes.v that is given something
   with variables in it, in the order of that file, a lemma named
   after the function, with the suffix [_renamed], says that on
   renamed arguments it gives the renamed answer. The lemmas are for
   the equation of the join of typing environments under a renaming
   (FlambdaEnvRenameJoinSpec.v), which needs the last of them,
   [n_way_join_renamed]: its conclusion is what that file asks of two
   joins of types that a renaming relates.

   The traversal is given the state of the join of environments, four
   functions of that state and four functions of other files. So the
   lemmas are in a section with two of each of them, but for the
   equality of block shapes, which reads no variable: the ones that
   the traversal of the types is given, and the ones that the
   traversal of the renamed types is given. What is supposed of each
   pair is that the second, on renamed arguments, gives the renamed
   answer of the first. The action of a renaming on the state is a
   variable of the section too.

   A list of things with the indices of the environments that they are
   of is renamed by [Keyed.map_values]: an index is not a name. A
   function type, a tag, a function slot, a value slot and a code id
   are left as they are by a renaming (FlambdaTypeRenameSpec.v), so
   the functions that read nothing else have no lemma:
   [n_way_join_function_type], [gather_function_types],
   [n_way_join_function_types] and the two before the change,
   [join_alloc_mode], [numbers_union], [strings_union],
   [n_way_join_index] and [n_way_join_block_shape].

   No function of the traversal orders, numbers or makes a variable.
   It gives the variables of the relations of a head to the join of
   [Simple]s and reads the answer by its constructor; everything else
   that it does with a variable goes through the functions that it is
   given.

   What is defined here is the action of a renaming on the types that
   FlambdaJoinTypes.v defines for itself: [rename_variant_extensions],
   [rename_variant_fields] and [rename_variant_parts], on what its
   fold over variants reads and gathers; and [rename_case], on a case
   of a row-like structure, whose index holds no variable. Its
   expanded types are those of FlambdaExpandHead.v, and the action on
   them is that of FlambdaProversRenameSpec.v ([rename_expanded_head],
   [rename_expanded]).

   The first section is about maps: what the operations of
   FlambdaMapSpec.v make of a map whose values have been through a
   function. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaNominalSpec FlambdaAlphaSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningBoolSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesRenamingProofs FlambdaTypingEnv FlambdaEnvRenameSpec
  FlambdaJoinEnv FlambdaJoinTypes FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaProversRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* Maps whose values have been through a function                           *)
(* ======================================================================== *)

Lemma find_values {B C : Type} (g : B -> C) (p : B -> bool) (p' : C -> bool)
    (l : list B) :
  (∀ b : B, p' (g b) = p b) -> find p' (map g l) = option_map g (find p l).
Proof.
  intros Hp. induction l as [ | b l IH]; simpl; [ reflexivity | ].
  rewrite Hp. destruct (p b); [ reflexivity | exact IH ].
Qed.

Section Values.

Context {K : Type} (eqb : K -> K -> bool).

Lemma keyed_map_values_cons {B C : Type} (g : B -> C) (key : K) (b : B)
    (m : Keyed.finite_map K B) :
  Keyed.map_values g ((key, b) :: m) = (key, g b) :: Keyed.map_values g m.
Proof. reflexivity. Qed.

Lemma keyed_map_values_rev {B C : Type} (g : B -> C)
    (m : Keyed.finite_map K B) :
  rev (Keyed.map_values g m) = Keyed.map_values g (rev m).
Proof. unfold Keyed.map_values. symmetry. apply map_rev. Qed.

Lemma keyed_forallb_values {B C : Type} (g : B -> C) (p : K * B -> bool)
    (p' : K * C -> bool) (m : Keyed.finite_map K B) :
  (∀ binding : K * B, p' (fst binding, g (snd binding)) = p binding) ->
  forallb p' (Keyed.map_values g m) = forallb p m.
Proof. apply forallb_renamed. Qed.

Lemma keyed_find_values {B C : Type} (g : B -> C) (p : K * B -> bool)
    (p' : K * C -> bool) (m : Keyed.finite_map K B) :
  (∀ binding : K * B, p' (fst binding, g (snd binding)) = p binding) ->
  find p' (Keyed.map_values g m)
  = option_map (fun binding => (fst binding, g (snd binding))) (find p m).
Proof. apply find_values. Qed.

Lemma keyed_filter_values {B C : Type} (g : B -> C) (p : K * B -> bool)
    (p' : K * C -> bool) (m : Keyed.finite_map K B) :
  (∀ binding : K * B, p' (fst binding, g (snd binding)) = p binding) ->
  filter p' (Keyed.map_values g m) = Keyed.map_values g (filter p m).
Proof. apply filter_renamed. Qed.

Lemma keyed_keys_values {B C : Type} (g : B -> C) (m : Keyed.finite_map K B) :
  map fst (Keyed.map_values g m) = map fst m.
Proof. unfold Keyed.map_values. rewrite map_map. reflexivity. Qed.

Lemma keyed_domain_values {B C : Type} (g : B -> C)
    (m : Keyed.finite_map K B) :
  Keyed.domain eqb (Keyed.map_values g m) = Keyed.domain eqb m.
Proof. unfold Keyed.domain. rewrite keyed_keys_values. reflexivity. Qed.

Lemma keyed_tabulate_values {B C : Type} (g : B -> C) (f : K -> option B)
    (f' : K -> option C) (keys : list K) :
  (∀ key : K, f' key = option_map g (f key)) ->
  Keyed.tabulate f' keys = Keyed.map_values g (Keyed.tabulate f keys).
Proof.
  intros Hf. unfold Keyed.tabulate, Keyed.map_values.
  induction keys as [ | key keys IH]; simpl; [ reflexivity | ].
  rewrite map_app, IH, Hf. destruct (f key); reflexivity.
Qed.

Lemma keyed_merge_values {A A' B B' C C' : Type} (ga : A -> A')
    (gb : B -> B') (gc : C -> C') (f : option A -> option B -> option C)
    (f' : option A' -> option B' -> option C') (m1 : Keyed.finite_map K A)
    (m2 : Keyed.finite_map K B) :
  (∀ (a : option A) (b : option B),
    f' (option_map ga a) (option_map gb b) = option_map gc (f a b)) ->
  Keyed.merge eqb f' (Keyed.map_values ga m1) (Keyed.map_values gb m2)
  = Keyed.map_values gc (Keyed.merge eqb f m1 m2).
Proof.
  intros Hf. unfold Keyed.merge. rewrite !keyed_keys_values.
  apply keyed_tabulate_values. intros key.
  rewrite !keyed_lookup_map_values. apply Hf.
Qed.

Lemma keyed_for_all_bindings_b_values {B C : Type} (g : B -> C)
    (p : K -> B -> bool) (p' : K -> C -> bool) (m : Keyed.finite_map K B) :
  (∀ (key : K) (b : B), p' key (g b) = p key b) ->
  Keyed.for_all_bindings_b eqb p' (Keyed.map_values g m)
  = Keyed.for_all_bindings_b eqb p m.
Proof.
  intros Hp. unfold Keyed.for_all_bindings_b, Keyed.map_values.
  generalize (@nil K) as seen.
  induction m as [ | [key b] m IH]; intros seen; simpl; [ reflexivity | ].
  rewrite Hp, IH. reflexivity.
Qed.

Lemma keyed_tabulate_fold_opt_values {B C S S' : Type} (g : B -> C)
    (gs : S -> S') (f : K -> S -> option (option B * S))
    (f' : K -> S' -> option (option C * S')) (keys : list K) :
  (∀ (key : K) (s : S),
    f' key (gs s)
    = option_map (fun result => (option_map g (fst result), gs (snd result)))
        (f key s)) ->
  ∀ s : S,
    Keyed.tabulate_fold_opt f' keys (gs s)
    = option_map
        (fun result => (Keyed.map_values g (fst result), gs (snd result)))
        (Keyed.tabulate_fold_opt f keys s).
Proof.
  intros Hf. induction keys as [ | key keys IH]; intros s; simpl;
    [ reflexivity | ].
  rewrite Hf. destruct (f key s) as [[b s1] | ]; simpl; [ | reflexivity ].
  rewrite IH.
  destruct (Keyed.tabulate_fold_opt f keys s1) as [[bindings s2] | ]; simpl;
    [ | reflexivity ].
  destruct b; reflexivity.
Qed.

Lemma keyed_filter_map_fold_values {A A' B B' S S' : Type} (ga : A -> A')
    (gb : B -> B') (gs : S -> S') (f : K -> A -> S -> option (option B * S))
    (f' : K -> A' -> S' -> option (option B' * S'))
    (m : Keyed.finite_map K A) :
  (∀ (key : K) (a : A) (s : S),
    f' key (ga a) (gs s)
    = option_map (fun result => (option_map gb (fst result), gs (snd result)))
        (f key a s)) ->
  ∀ s : S,
    Keyed.filter_map_fold eqb f' (Keyed.map_values ga m) (gs s)
    = option_map
        (fun result => (Keyed.map_values gb (fst result), gs (snd result)))
        (Keyed.filter_map_fold eqb f m s).
Proof.
  intros Hf s. unfold Keyed.filter_map_fold, Keyed.filter_map_fold_on.
  rewrite keyed_domain_values.
  apply keyed_tabulate_fold_opt_values. intros key s'.
  rewrite keyed_lookup_map_values.
  destruct (Keyed.lookup eqb key m) as [a | ]; simpl;
    [ apply Hf | reflexivity ].
Qed.

End Values.

(* The same for [merge] of the maps keyed by names. *)
Lemma merge_values {A A' B B' C C' : Type} (ga : A -> A') (gb : B -> B')
    (gc : C -> C') (f : option A -> option B -> option C)
    (f' : option A' -> option B' -> option C') (m1 : finite_map A)
    (m2 : finite_map B) :
  (∀ (a : option A) (b : option B),
    f' (option_map ga a) (option_map gb b) = option_map gc (f a b)) ->
  merge f' (map_values ga m1) (map_values gb m2)
  = map_values gc (merge f m1 m2).
Proof. exact (keyed_merge_values Pos.eqb ga gb gc f f' m1 m2). Qed.

Lemma for_all_bindings_b_values {B C : Type} (g : B -> C)
    (p : positive -> B -> bool) (p' : positive -> C -> bool)
    (m : finite_map B) :
  (∀ (key : positive) (b : B), p' key (g b) = p key b) ->
  for_all_bindings_b p' (map_values g m) = for_all_bindings_b p m.
Proof. exact (keyed_for_all_bindings_b_values Pos.eqb g p p' m). Qed.

(* ======================================================================== *)
(* The expanded types                                                       *)
(* ======================================================================== *)

(* The action of a renaming on the expanded types is
   FlambdaProversRenameSpec.rename_expanded: the head is renamed, the
   kind is kept. The expanded types of FlambdaJoinTypes.v are those of
   FlambdaExpandHead.v, which that action is defined on. *)

Lemma create_bottom_renamed (r : renaming) (kind : flambda_kind) :
  rename_expanded r (Expanded_type.create_bottom kind)
  = Expanded_type.create_bottom kind.
Proof. reflexivity. Qed.

Lemma create_unknown_renamed (r : renaming) (kind : flambda_kind) :
  rename_expanded r (Expanded_type.create_unknown kind)
  = Expanded_type.create_unknown kind.
Proof. reflexivity. Qed.

Lemma create_value_renamed (r : renaming) (head : head_of_kind_value) :
  rename_expanded r (Expanded_type.create_value head)
  = Expanded_type.create_value (rename_head_of_kind_value r head).
Proof. reflexivity. Qed.

Lemma create_naked_immediate_renamed (r : renaming)
    (head : head_of_kind_naked_immediate) :
  rename_expanded r (Expanded_type.create_naked_immediate head)
  = Expanded_type.create_naked_immediate
      (rename_head_of_kind_naked_immediate r head).
Proof. reflexivity. Qed.

Lemma create_naked_number_renamed (r : renaming) (k : naked_number_kind)
    (head : list Z) :
  rename_expanded r (Expanded_type.create_naked_number k head)
  = Expanded_type.create_naked_number k head.
Proof. reflexivity. Qed.

Lemma create_rec_info_renamed (r : renaming) (head : unit) :
  rename_expanded r (Expanded_type.create_rec_info head)
  = Expanded_type.create_rec_info head.
Proof. reflexivity. Qed.

Lemma create_region_renamed (r : renaming) (head : unit) :
  rename_expanded r (Expanded_type.create_region head)
  = Expanded_type.create_region head.
Proof. reflexivity. Qed.

Lemma to_type_renamed (r : renaming) (expanded : Expanded_type.t) :
  Expanded_type.to_type (rename_expanded r expanded)
  = rename_type r (Expanded_type.to_type expanded).
Proof.
  destruct expanded as [k [ | [head | head | k' head | head | head] | ]];
    unfold Expanded_type.to_type, rename_expanded; simpl;
    rewrite ?rename_type_unknown, ?rename_type_bottom; reflexivity.
Qed.

Lemma unknown_like_renamed (r : renaming) (expanded : Expanded_type.t) :
  Expanded_type.unknown_like (rename_expanded r expanded)
  = rename_expanded r (Expanded_type.unknown_like expanded).
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* What the traversal is made with                                          *)
(* ======================================================================== *)

Lemma untagged_const_bool_renamed (r : renaming) (b : bool) :
  rename_simple r (untagged_const_bool b) = untagged_const_bool b.
Proof. reflexivity. Qed.

Lemma untagged_const_int_renamed (r : renaming) (i : Z) :
  rename_simple r (untagged_const_int i) = untagged_const_int i.
Proof. reflexivity. Qed.

Lemma row_like_for_blocks_all_tags_renamed (r : renaming)
    (blocks : row_like_for_blocks) :
  row_like_for_blocks_all_tags (rename_row_like_for_blocks r blocks)
  = row_like_for_blocks_all_tags blocks.
Proof.
  destruct blocks as [known_tags [other | ] alloc_mode]; simpl;
    [ reflexivity | ].
  rewrite keyed_domain_values. reflexivity.
Qed.

Lemma deduce_is_int_renamed (r : renaming)
    (blocks : Or_unknown.t row_like_for_blocks)
    (imms : Or_unknown.t flambda_type) :
  deduce_is_int (or_unknown_map (rename_row_like_for_blocks r) blocks)
    (or_unknown_map (rename_type r) imms)
  = deduce_is_int blocks imms.
Proof.
  unfold deduce_is_int.
  destruct blocks as [blocks | ], imms as [imms | ];
    cbn [or_unknown_map];
    rewrite ?row_like_for_blocks_is_bottom_renamed,
      ?is_obviously_bottom_renamed;
    reflexivity.
Qed.

Lemma deduce_is_int_simple_renamed (r : renaming)
    (blocks : Or_unknown.t row_like_for_blocks)
    (imms : Or_unknown.t flambda_type) (is_int_var : option variable) :
  deduce_is_int_simple (or_unknown_map (rename_row_like_for_blocks r) blocks)
    (or_unknown_map (rename_type r) imms)
    (option_map (rename_variable r) is_int_var)
  = option_map (rename_simple r)
      (deduce_is_int_simple blocks imms is_int_var).
Proof.
  unfold deduce_is_int_simple. rewrite deduce_is_int_renamed.
  destruct (deduce_is_int blocks imms) as [is_int | ],
    is_int_var as [var | ];
    reflexivity.
Qed.

Lemma deduce_get_tag_renamed (r : renaming)
    (blocks : Or_unknown.t row_like_for_blocks) :
  deduce_get_tag (or_unknown_map (rename_row_like_for_blocks r) blocks)
  = deduce_get_tag blocks.
Proof.
  destruct blocks as [blocks | ]; cbn [or_unknown_map deduce_get_tag];
    [ | reflexivity ].
  rewrite row_like_for_blocks_all_tags_renamed. reflexivity.
Qed.

Lemma deduce_get_tag_simple_renamed (r : renaming)
    (blocks : Or_unknown.t row_like_for_blocks)
    (get_tag_var : option variable) :
  deduce_get_tag_simple
    (or_unknown_map (rename_row_like_for_blocks r) blocks)
    (option_map (rename_variable r) get_tag_var)
  = or_unknown_or_bottom_map (rename_simple r)
      (deduce_get_tag_simple blocks get_tag_var).
Proof.
  unfold deduce_get_tag_simple. rewrite deduce_get_tag_renamed.
  destruct (deduce_get_tag blocks) as [ | get_tag | ],
    get_tag_var as [var | ];
    reflexivity.
Qed.

Lemma known_contents_renamed {A B : Type} (g : A -> B)
    (acc : list (Index.t * A))
    (or_unknowns : list (Index.t * Or_unknown.t A)) :
  known_contents (Keyed.map_values g acc)
    (Keyed.map_values (or_unknown_map g) or_unknowns)
  = option_map (Keyed.map_values g) (known_contents acc or_unknowns).
Proof.
  revert acc.
  induction or_unknowns as [ | [id [contents | ]] or_unknowns IH];
    intros acc; [ reflexivity | | reflexivity ].
  exact (IH ((id, contents) :: acc)).
Qed.

Lemma is_null_simples_renamed (r : renaming)
    (heads : list (Index.t * head_of_kind_value)) :
  is_null_simples (Keyed.map_values (rename_head_of_kind_value r) heads)
  = option_map (Keyed.map_values (rename_simple r)) (is_null_simples heads).
Proof.
  induction heads as [ | [id [non_null [ | [is_null | ]]]] heads IH];
    [ reflexivity | | | reflexivity ];
    rewrite keyed_map_values_cons;
    cbn [is_null_simples rename_head_of_kind_value rename_is_null
           option_map];
    rewrite IH; destruct (is_null_simples heads); reflexivity.
Qed.

Lemma unknown_from_shape_renamed (r : renaming) (shape : block_shape)
    (index : nat) :
  option_map (rename_type r) (FlambdaJoinTypes.unknown_from_shape shape index)
  = FlambdaJoinTypes.unknown_from_shape shape index.
Proof.
  unfold FlambdaJoinTypes.unknown_from_shape.
  destruct (block_shape_element_kind shape index) as [k | ]; simpl;
    [ rewrite rename_type_unknown | ]; reflexivity.
Qed.

Lemma env_extension_is_empty_renamed (r : renaming)
    (extension : env_extension) :
  env_extension_is_empty (rename_env_extension r extension)
  = env_extension_is_empty extension.
Proof. destruct extension as [[ | equation equations]]; reflexivity. Qed.

Lemma closures_entry_function_types_renamed (r : renaming)
    (entry : closures_entry) :
  closures_entry_function_types (rename_closures_entry r entry)
  = closures_entry_function_types entry.
Proof. destruct entry; reflexivity. Qed.

Lemma closures_entry_closure_types_renamed (r : renaming)
    (entry : closures_entry) :
  closures_entry_closure_types (rename_closures_entry r entry)
  = map_values (rename_type r) (closures_entry_closure_types entry).
Proof. destruct entry; reflexivity. Qed.

Lemma closures_entry_value_slot_types_renamed (r : renaming)
    (entry : closures_entry) :
  closures_entry_value_slot_types (rename_closures_entry r entry)
  = map_values (rename_type r) (closures_entry_value_slot_types entry).
Proof. destruct entry; reflexivity. Qed.

Lemma row_like_known_tags_renamed (r : renaming)
    (blocks : row_like_for_blocks) :
  row_like_known_tags (rename_row_like_for_blocks r blocks)
  = Keyed.map_values
      (or_unknown_map
         (rename_block_case (rename_type r) (rename_env_extension r)))
      (row_like_known_tags blocks).
Proof. destruct blocks; reflexivity. Qed.

Lemma row_like_other_tags_renamed (r : renaming)
    (blocks : row_like_for_blocks) :
  row_like_other_tags (rename_row_like_for_blocks r blocks)
  = or_bottom_map
      (rename_block_case (rename_type r) (rename_env_extension r))
      (row_like_other_tags blocks).
Proof. destruct blocks; reflexivity. Qed.

Lemma row_like_alloc_mode_renamed (r : renaming)
    (blocks : row_like_for_blocks) :
  row_like_alloc_mode (rename_row_like_for_blocks r blocks)
  = row_like_alloc_mode blocks.
Proof. destruct blocks; reflexivity. Qed.

Lemma row_like_known_closures_renamed (r : renaming)
    (closures : row_like_for_closures) :
  row_like_known_closures (rename_row_like_for_closures r closures)
  = map_values
      (rename_closures_case (rename_closures_entry r)
         (rename_env_extension r))
      (row_like_known_closures closures).
Proof. destruct closures; reflexivity. Qed.

Lemma gather_components_renamed {A B : Type} (g : A -> B)
    (indexed_products : list (Index.t * finite_map A)) :
  gather_components (Keyed.map_values (map_values g) indexed_products)
  = map_values (Keyed.map_values g) (gather_components indexed_products).
Proof.
  destruct indexed_products as [ | [id1 indexed_product1] indexed_products];
    [ reflexivity | ].
  rewrite keyed_map_values_cons. unfold gather_components.
  replace
    (map_values (fun val1 : B => [(id1, val1)])
       (map_values g indexed_product1))
    with
      (map_values (Keyed.map_values g)
         (map_values (fun val1 : A => [(id1, val1)]) indexed_product1))
    by (unfold map_values; rewrite !map_map; reflexivity).
  apply
    (fold_left_renamed (map_values (Keyed.map_values g))
       (fun indexed_product : Index.t * finite_map A =>
          (fst indexed_product, map_values g (snd indexed_product)))).
  intros components_by_index indexed_product2. cbn [fst snd].
  apply merge_values. intros [vals | ] [val2 | ]; reflexivity.
Qed.

Section Merge_map_known_renamed.

Context {K value value' other other' : Type} (eqb : K -> K -> bool)
  (gv : value -> value') (go : other -> other') (of_other : other -> value)
  (of_other' : other' -> value').

Hypothesis of_other_related :
  ∀ known : other, of_other' (go known) = gv (of_other known).

Lemma add_other_renamed (acc : list (Index.t * value)) (id : Index.t)
    (other_or_bottom : Or_bottom.t other) :
  add_other of_other' (Keyed.map_values gv acc) id
    (or_bottom_map go other_or_bottom)
  = Keyed.map_values gv (add_other of_other acc id other_or_bottom).
Proof.
  destruct other_or_bottom as [known | ]; cbn [or_bottom_map add_other];
    [ | reflexivity ].
  rewrite of_other_related. reflexivity.
Qed.

Lemma gather_known_renamed
    (maps :
       list (Index.t * (Keyed.finite_map K value * Or_bottom.t other))) :
  gather_known eqb of_other'
    (Keyed.map_values
       (fun map_and_other : Keyed.finite_map K value * Or_bottom.t other =>
          (Keyed.map_values gv (fst map_and_other),
           or_bottom_map go (snd map_and_other)))
       maps)
  = Keyed.map_values (Keyed.map_values gv) (gather_known eqb of_other maps).
Proof.
  destruct maps as [ | [id1 [map1 other1]] maps]; [ reflexivity | ].
  rewrite keyed_map_values_cons. unfold gather_known. cbn [fst snd].
  pose
    (g_acc :=
       fun acc :
             Keyed.finite_map K (list (Index.t * value))
             * list (Index.t * value) =>
         (Keyed.map_values (Keyed.map_values gv) (fst acc),
          Keyed.map_values gv (snd acc))).
  pose
    (g_input :=
       fun input : Index.t * (Keyed.finite_map K value * Or_bottom.t other) =>
         (fst input,
          (Keyed.map_values gv (fst (snd input)),
           or_bottom_map go (snd (snd input))))).
  match goal with
  | |- fst (List.fold_left ?f' _ _) = _ (fst (List.fold_left ?f _ ?init)) =>
    transitivity (fst (List.fold_left f' (map g_input maps) (g_acc init)));
      [ | rewrite (fold_left_renamed g_acc g_input f f');
          [ reflexivity | ] ]
  end.
  - f_equal. f_equal. unfold g_acc. cbn [fst snd]. f_equal.
    + unfold Keyed.map_values. rewrite !map_map. reflexivity.
    + apply (add_other_renamed [] id1 other1).
  - intros acc input. unfold g_acc, g_input. cbn [fst snd].
    rewrite add_other_renamed. f_equal.
    apply keyed_merge_values.
    intros [vals | ] [val2 | ]; cbn [option_map]; try reflexivity.
    rewrite add_other_renamed. reflexivity.
Qed.

End Merge_map_known_renamed.

Lemma all_known_renamed {A B : Type} (g : A -> B)
    (or_unknowns : list (Index.t * Or_unknown.t A)) :
  all_known (Keyed.map_values (or_unknown_map g) or_unknowns)
  = option_map (Keyed.map_values g) (all_known or_unknowns).
Proof.
  induction or_unknowns as [ | [id [a | ]] or_unknowns IH];
    [ reflexivity | | reflexivity ].
  rewrite keyed_map_values_cons. cbn [all_known or_unknown_map].
  rewrite IH. destruct (all_known or_unknowns); reflexivity.
Qed.

Lemma non_bottom_renamed {A B : Type} (g : A -> B) (acc : list (Index.t * A))
    (or_bottoms : list (Index.t * Or_bottom.t A)) :
  non_bottom (Keyed.map_values g acc)
    (Keyed.map_values (or_bottom_map g) or_bottoms)
  = Keyed.map_values g (non_bottom acc or_bottoms).
Proof.
  revert acc.
  induction or_bottoms as [ | [id [a | ]] or_bottoms IH]; intros acc;
    [ reflexivity | | exact (IH acc) ].
  exact (IH ((id, a) :: acc)).
Qed.

(* The action of a renaming on the extensions of a variant, as
   [rename_type] has it. *)
Definition rename_variant_extensions (r : renaming)
    (extensions : variant_extensions) : variant_extensions :=
  match extensions with
  | No_extensions => No_extensions
  | Ext when_immediate when_block =>
    Ext (rename_env_extension r when_immediate)
      (rename_env_extension r when_block)
  end.

Lemma gather_extensions_renamed (r : renaming)
    (when_immediates when_blocks : list (Index.t * env_extension))
    (extensions : list (Index.t * variant_extensions)) :
  gather_extensions
    (Keyed.map_values (rename_env_extension r) when_immediates)
    (Keyed.map_values (rename_env_extension r) when_blocks)
    (Keyed.map_values (rename_variant_extensions r) extensions)
  = option_map
      (fun gathered =>
         (Keyed.map_values (rename_env_extension r) (fst gathered),
          Keyed.map_values (rename_env_extension r) (snd gathered)))
      (gather_extensions when_immediates when_blocks extensions).
Proof.
  revert when_immediates when_blocks.
  induction extensions
    as [ | [id [ | when_immediate when_block]] extensions IH];
    intros when_immediates when_blocks; [ reflexivity | reflexivity | ].
  exact
    (IH ((id, when_immediate) :: when_immediates)
       ((id, when_block) :: when_blocks)).
Qed.

Lemma non_null_heads_renamed (r : renaming)
    (acc : list (Index.t * head_of_kind_value_non_null))
    (heads : list (Index.t * head_of_kind_value)) :
  non_null_heads
    (Keyed.map_values (rename_head_of_kind_value_non_null r) acc)
    (Keyed.map_values (rename_head_of_kind_value r) heads)
  = option_map (Keyed.map_values (rename_head_of_kind_value_non_null r))
      (non_null_heads acc heads).
Proof.
  revert acc.
  induction heads as [ | [id2 [[ | head | ] is_null]] heads IH]; intros acc;
    [ reflexivity | reflexivity | | exact (IH acc) ].
  exact (IH ((id2, head) :: acc)).
Qed.

Lemma gather_heads_renamed (r : renaming) {A A' B B' : Type} (ga : A -> A')
    (gb : B -> B') (read : head_of_kind_value_non_null -> option A)
    (read' : head_of_kind_value_non_null -> option A')
    (combine : Index.t -> A -> B -> B) (combine' : Index.t -> A' -> B' -> B')
    (other_heads : list (Index.t * head_of_kind_value_non_null)) (init : B) :
  (∀ head : head_of_kind_value_non_null,
    read' (rename_head_of_kind_value_non_null r head)
    = option_map ga (read head)) ->
  (∀ (id : Index.t) (a : A) (b : B),
    combine' id (ga a) (gb b) = gb (combine id a b)) ->
  gather_heads read' combine'
    (Keyed.map_values (rename_head_of_kind_value_non_null r) other_heads)
    (gb init)
  = option_map gb (gather_heads read combine other_heads init).
Proof.
  intros Hread Hcombine.
  induction other_heads as [ | [other_id other_head] other_heads IH];
    [ reflexivity | ].
  rewrite keyed_map_values_cons. cbn [gather_heads]. rewrite IH.
  destruct (gather_heads read combine other_heads init) as [acc | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite Hread. destruct (read other_head) as [a | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite Hcombine. reflexivity.
Qed.

(* The action of a renaming on the fields of a variant and on what the
   fold over variants gathers. *)
Definition rename_variant_fields (r : renaming) (fields : variant_fields)
    : variant_fields :=
  {| variant_is_int := option_map (rename_variable r) (variant_is_int fields);
     variant_immediates :=
       or_unknown_map (rename_type r) (variant_immediates fields);
     variant_get_tag :=
       option_map (rename_variable r) (variant_get_tag fields);
     variant_blocks :=
       or_unknown_map (rename_row_like_for_blocks r) (variant_blocks fields);
     variant_extensions_of :=
       rename_variant_extensions r (variant_extensions_of fields);
     variant_is_unique := variant_is_unique fields |}.

Definition rename_variant_parts (r : renaming) (parts : variant_parts)
    : variant_parts :=
  {| parts_blocks :=
       Keyed.map_values (or_unknown_map (rename_row_like_for_blocks r))
         (parts_blocks parts);
     parts_immediates :=
       Keyed.map_values (or_unknown_map (rename_type r))
         (parts_immediates parts);
     parts_extensions :=
       Keyed.map_values (rename_variant_extensions r)
         (parts_extensions parts);
     parts_is_unique := parts_is_unique parts;
     parts_is_int :=
       option_map (Keyed.map_values (rename_simple r)) (parts_is_int parts);
     parts_get_tag :=
       option_map (Keyed.map_values (rename_simple r)) (parts_get_tag parts)
  |}.

Lemma read_variant_renamed (r : renaming)
    (head : head_of_kind_value_non_null) :
  read_variant (rename_head_of_kind_value_non_null r head)
  = option_map (rename_variant_fields r) (read_variant head).
Proof. destruct head; reflexivity. Qed.

Lemma first_variant_parts_renamed (r : renaming) (first_id : Index.t)
    (first : variant_fields) :
  first_variant_parts first_id (rename_variant_fields r first)
  = rename_variant_parts r (first_variant_parts first_id first).
Proof.
  unfold first_variant_parts, rename_variant_parts, rename_variant_fields.
  cbn [variant_is_int variant_immediates variant_get_tag variant_blocks
         variant_extensions_of variant_is_unique parts_blocks
         parts_immediates parts_extensions parts_is_unique parts_is_int
         parts_get_tag].
  rewrite deduce_is_int_simple_renamed, deduce_get_tag_simple_renamed.
  destruct
    (deduce_is_int_simple (variant_blocks first) (variant_immediates first)
       (variant_is_int first)),
    (deduce_get_tag_simple (variant_blocks first) (variant_get_tag first));
    reflexivity.
Qed.

Lemma add_variant_parts_renamed (r : renaming) (other_id : Index.t)
    (other : variant_fields) (acc : variant_parts) :
  add_variant_parts other_id (rename_variant_fields r other)
    (rename_variant_parts r acc)
  = rename_variant_parts r (add_variant_parts other_id other acc).
Proof.
  unfold add_variant_parts, rename_variant_parts, rename_variant_fields.
  cbn [variant_is_int variant_immediates variant_get_tag variant_blocks
         variant_extensions_of variant_is_unique parts_blocks
         parts_immediates parts_extensions parts_is_unique parts_is_int
         parts_get_tag].
  rewrite deduce_is_int_simple_renamed, deduce_get_tag_simple_renamed.
  destruct (parts_is_int acc), (parts_get_tag acc),
    (deduce_is_int_simple (variant_blocks other) (variant_immediates other)
       (variant_is_int other)),
    (deduce_get_tag_simple (variant_blocks other) (variant_get_tag other));
    reflexivity.
Qed.

Lemma read_mutable_block_renamed (r : renaming)
    (head : head_of_kind_value_non_null) :
  read_mutable_block (rename_head_of_kind_value_non_null r head)
  = read_mutable_block head.
Proof. destruct head; reflexivity. Qed.

Lemma read_boxed_number_renamed (r : renaming) (k : boxable_number)
    (head : head_of_kind_value_non_null) :
  read_boxed_number k (rename_head_of_kind_value_non_null r head)
  = option_map
      (fun contents : flambda_type * Alloc_mode.t =>
         (rename_type r (fst contents), snd contents))
      (read_boxed_number k head).
Proof.
  destruct head as [ | | k' n alloc_mode | | ]; try reflexivity.
  cbn [rename_head_of_kind_value_non_null read_boxed_number].
  destruct (kind_eqb (unboxed_kind k) (unboxed_kind k')); reflexivity.
Qed.

Lemma read_closures_renamed (r : renaming)
    (head : head_of_kind_value_non_null) :
  read_closures (rename_head_of_kind_value_non_null r head)
  = option_map
      (fun closures : row_like_for_closures * Alloc_mode.t =>
         (rename_row_like_for_closures r (fst closures), snd closures))
      (read_closures head).
Proof. destruct head; reflexivity. Qed.

Lemma read_string_renamed (r : renaming)
    (head : head_of_kind_value_non_null) :
  read_string (rename_head_of_kind_value_non_null r head) = read_string head.
Proof. destruct head; reflexivity. Qed.

Lemma extract_heads_renamed (r : renaming) {A : Type} (ga : A -> A)
    (extract extract' : Expanded_type.head -> option A)
    (expandeds : list (Index.t * Expanded_type.t)) :
  (∀ head : Expanded_type.head,
    extract' (rename_expanded_head r head) = option_map ga (extract head)) ->
  extract_heads extract' (Keyed.map_values (rename_expanded r) expandeds)
  = option_map (option_map (Keyed.map_values ga))
      (extract_heads extract expandeds).
Proof.
  intros Hextract.
  induction expandeds as [ | [id2 [k [ | head2 | ]]] expandeds IH];
    [ reflexivity | reflexivity | | exact IH ].
  rewrite keyed_map_values_cons.
  cbn [extract_heads rename_expanded Expanded_type.descr Expanded_type.kind
         or_unknown_or_bottom_map].
  rewrite Hextract, IH.
  destruct (extract head2) as [head | ]; cbn [option_map]; [ | reflexivity ].
  destruct (extract_heads extract expandeds) as [[heads | ] | ]; reflexivity.
Qed.

Lemma alias_simples_renamed (r : renaming)
    (ts : list (Index.t * flambda_type)) :
  alias_simples (Keyed.map_values (rename_type r) ts)
  = option_map (Keyed.map_values (rename_simple r)) (alias_simples ts).
Proof.
  induction ts as [ | [id ty] ts IH]; [ reflexivity | ].
  rewrite keyed_map_values_cons. cbn [alias_simples].
  rewrite get_alias_opt_renamed, IH.
  destruct (get_alias_opt ty) as [simple | ]; cbn [option_map];
    [ | reflexivity ].
  destruct (alias_simples ts); reflexivity.
Qed.

Lemma fields_at_renamed (r : renaming) (index : nat)
    (fields : list (Index.t * list flambda_type)) :
  fields_at index (Keyed.map_values (map (rename_type r)) fields)
  = option_map (Keyed.map_values (rename_type r)) (fields_at index fields).
Proof.
  induction fields as [ | [id fields_of_id] fields IH]; [ reflexivity | ].
  rewrite keyed_map_values_cons. cbn [fields_at].
  rewrite nth_error_map, IH.
  destruct (nth_error fields_of_id index) as [field | ]; cbn [option_map];
    [ | reflexivity ].
  destruct (fields_at index fields); reflexivity.
Qed.

(* ======================================================================== *)
(* The traversal                                                            *)
(* ======================================================================== *)

(* [replace_by_equation lhs H]: the equation [H] rewrites [lhs], which
   is its left side up to conversion. *)
Local Ltac replace_by_equation lhs H :=
  match type of H with
  | _ = ?rhs => replace lhs with rhs by (symmetry; exact H)
  end.

Section Traversal_renamed.

Variable r : renaming.

(* The state of the join of environments and the action of the
   renaming on it; what the traversal of the types is given, and what
   the traversal of the renamed types is given. *)
Context {state : Type} (rename_state : state -> state)
  (joined_env joined_env' : state -> Index.t -> option Typing_env.t)
  (n_way_join_simples n_way_join_simples' :
     state -> flambda_kind -> list (Index.t * simple)
     -> option (Or_bottom.t simple * state))
  (import_type import_type' : state -> flambda_type -> option state)
  (join_env_extension join_env_extension' :
     state -> list (Index.t * env_extension)
     -> option (Or_bottom.t (env_extension * state)))
  (expand_head expand_head' :
     Typing_env.t -> flambda_type -> option Expanded_type.t)
  (reduce reduce' :
     Typing_env.t -> head_of_kind_naked_immediate
     -> Or_bottom.t head_of_kind_naked_immediate)
  (type_eqb type_eqb' : flambda_type -> flambda_type -> bool)
  (block_shape_eqb : block_shape -> block_shape -> bool).

(* The action on an answer that comes with the state. *)
Local Abbreviation with_state f :=
  (fun result => (f (fst result), rename_state (snd result))).

(* On renamed arguments, each function given to the traversal of the
   renamed types gives the renamed answer. *)
Hypothesis joined_env_related :
  ∀ (env : state) (index : Index.t),
    joined_env' (rename_state env) index
    = option_map (rename_typing_env r) (joined_env env index).

Hypothesis n_way_join_simples_related :
  ∀ (env : state) (kind : flambda_kind) (simples : list (Index.t * simple)),
    n_way_join_simples' (rename_state env) kind
      (Keyed.map_values (rename_simple r) simples)
    = option_map
        (fun result =>
           (or_bottom_map (rename_simple r) (fst result),
            rename_state (snd result)))
        (n_way_join_simples env kind simples).

Hypothesis import_type_related :
  ∀ (env : state) (ty : flambda_type),
    import_type' (rename_state env) (rename_type r ty)
    = option_map rename_state (import_type env ty).

Hypothesis join_env_extension_related :
  ∀ (env : state) (extensions : list (Index.t * env_extension)),
    join_env_extension' (rename_state env)
      (Keyed.map_values (rename_env_extension r) extensions)
    = option_map
        (or_bottom_map
           (fun result =>
              (rename_env_extension r (fst result),
               rename_state (snd result))))
        (join_env_extension env extensions).

Hypothesis expand_head_related :
  ∀ (tenv : Typing_env.t) (t : flambda_type),
    expand_head' (rename_typing_env r tenv) (rename_type r t)
    = option_map (rename_expanded r) (expand_head tenv t).

Hypothesis reduce_related :
  ∀ (tenv : Typing_env.t) (head : head_of_kind_naked_immediate),
    reduce' (rename_typing_env r tenv)
      (rename_head_of_kind_naked_immediate r head)
    = or_bottom_map (rename_head_of_kind_naked_immediate r)
        (reduce tenv head).

Hypothesis type_eqb_related :
  ∀ t t' : flambda_type,
    type_eqb' (rename_type r t) (rename_type r t') = type_eqb t t'.

Lemma map_join_result_renamed {A A' B B' : Type} (ga : A -> A')
    (gb : B -> B') (f : A -> B) (f' : A' -> B')
    (result : option (Or_unknown.t A * state)) :
  (∀ a : A, f' (ga a) = gb (f a)) ->
  map_join_result f' (option_map (with_state (or_unknown_map ga)) result)
  = option_map (with_state (or_unknown_map gb)) (map_join_result f result).
Proof.
  intros Hf. destruct result as [[[a | ] env] | ]; try reflexivity.
  cbn [option_map map_join_result fst snd or_unknown_map
         FlambdaJoinTypes.or_unknown_map].
  rewrite Hf. reflexivity.
Qed.

Lemma n_way_join_unknown_renamed {A A' B B' : Type} (ga : A -> A')
    (gb : B -> B')
    (n_way_join_contents :
       state -> list (Index.t * A) -> option (Or_unknown.t B * state))
    (n_way_join_contents' :
       state -> list (Index.t * A') -> option (Or_unknown.t B' * state)) :
  (∀ (env : state) (contents : list (Index.t * A)),
    n_way_join_contents' (rename_state env) (Keyed.map_values ga contents)
    = option_map (with_state (or_unknown_map gb))
        (n_way_join_contents env contents)) ->
  ∀ (env : state) (or_unknowns : list (Index.t * Or_unknown.t A)),
    n_way_join_unknown n_way_join_contents' (rename_state env)
      (Keyed.map_values (or_unknown_map ga) or_unknowns)
    = option_map (with_state (or_unknown_map gb))
        (n_way_join_unknown n_way_join_contents env or_unknowns).
Proof.
  intros Hcontents env or_unknowns. unfold n_way_join_unknown.
  rewrite
    (known_contents_renamed ga [] or_unknowns
     : known_contents [] (Keyed.map_values (or_unknown_map ga) or_unknowns)
       = _).
  destruct (known_contents [] or_unknowns) as [contents | ];
    cbn [option_map]; [ apply Hcontents | reflexivity ].
Qed.

Lemma n_way_join_head_of_kind_naked_number_renamed (env : state)
    (t1 : list Z * Index.t) (ts : list (Index.t * list Z)) :
  n_way_join_head_of_kind_naked_number (rename_state env) t1 ts
  = option_map (with_state (or_unknown_map (fun numbers => numbers)))
      (n_way_join_head_of_kind_naked_number env t1 ts).
Proof. reflexivity. Qed.

Lemma n_way_join_relation_simples_renamed (env : state)
    (simples_opt : option (list (Index.t * simple))) :
  n_way_join_relation_simples n_way_join_simples' (rename_state env)
    (option_map (Keyed.map_values (rename_simple r)) simples_opt)
  = option_map (with_state (option_map (rename_variable r)))
      (n_way_join_relation_simples n_way_join_simples env simples_opt).
Proof.
  destruct simples_opt as [simples | ];
    cbn [option_map n_way_join_relation_simples]; [ | reflexivity ].
  rewrite n_way_join_simples_related.
  destruct
    (n_way_join_simples env (Naked_number Kind_naked_immediate) simples)
    as [[[[var | sym | c] | ] env'] | ];
    reflexivity.
Qed.

Lemma n_way_join_is_null_renamed (env : state)
    (heads : list (Index.t * head_of_kind_value)) :
  n_way_join_is_null n_way_join_simples' (rename_state env)
    (Keyed.map_values (rename_head_of_kind_value r) heads)
  = option_map (with_state (rename_is_null r))
      (n_way_join_is_null n_way_join_simples env heads).
Proof.
  unfold n_way_join_is_null. rewrite is_null_simples_renamed.
  destruct (is_null_simples heads) as [simples | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite n_way_join_simples_related.
  destruct
    (n_way_join_simples env (Naked_number Kind_naked_immediate) simples)
    as [[[[var | sym | c] | ] env'] | ];
    try reflexivity.
  destruct c; try reflexivity.
  cbn [option_map or_bottom_map rename_simple fst snd].
  destruct (Z.eqb _ 0); reflexivity.
Qed.

Lemma join_naked_immediates_renamed (env : state)
    (immediates : Or_unknown.t (list Z))
    (heads : list (Index.t * head_of_kind_naked_immediate)) :
  join_naked_immediates joined_env' reduce' (rename_state env) immediates
    (Keyed.map_values (rename_head_of_kind_naked_immediate r) heads)
  = join_naked_immediates joined_env reduce env immediates heads.
Proof.
  revert immediates.
  induction heads as [ | [id2 head2] heads IH]; intros immediates;
    [ reflexivity | ].
  rewrite keyed_map_values_cons. cbn [join_naked_immediates].
  rewrite joined_env_related.
  destruct (joined_env env id2) as [tenv | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite reduce_related.
  destruct (reduce tenv head2) as [reduced | ]; cbn [or_bottom_map];
    apply IH.
Qed.

Lemma n_way_join_head_of_kind_naked_immediate_renamed (env : state)
    (heads : list (Index.t * head_of_kind_naked_immediate)) :
  n_way_join_head_of_kind_naked_immediate joined_env' reduce'
    (rename_state env)
    (Keyed.map_values (rename_head_of_kind_naked_immediate r) heads)
  = option_map
      (with_state (or_unknown_map (rename_head_of_kind_naked_immediate r)))
      (n_way_join_head_of_kind_naked_immediate joined_env reduce env heads).
Proof.
  unfold n_way_join_head_of_kind_naked_immediate.
  rewrite join_naked_immediates_renamed.
  destruct
    (join_naked_immediates joined_env reduce env (Or_unknown.Known []) heads)
    as [[imms | ] | ];
    try reflexivity.
  destruct (Keyed.is_empty imms); reflexivity.
Qed.

Lemma n_way_join_head_of_kind_naked_numbers_renamed (env : state)
    (t1 : list Z * Index.t) (ts : list (Index.t * list Z)) :
  n_way_join_head_of_kind_naked_numbers (rename_state env) t1 ts
  = option_map (with_state (or_unknown_map (fun numbers => numbers)))
      (n_way_join_head_of_kind_naked_numbers env t1 ts).
Proof. reflexivity. Qed.

Lemma n_way_join_head_of_kind_rec_info_renamed (env : state)
    (t1 : unit * Index.t) (ts : list (Index.t * unit)) :
  n_way_join_head_of_kind_rec_info (rename_state env) t1 ts
  = option_map (with_state (or_unknown_map (fun head => head)))
      (n_way_join_head_of_kind_rec_info env t1 ts).
Proof. reflexivity. Qed.

Lemma n_way_join_head_of_kind_region_renamed (env : state)
    (t1 : unit * Index.t) (ts : list (Index.t * unit)) :
  n_way_join_head_of_kind_region (rename_state env) t1 ts
  = option_map (with_state (or_unknown_map (fun head => head)))
      (n_way_join_head_of_kind_region env t1 ts).
Proof. reflexivity. Qed.

(* [n_way_join_env_extension] of the traversal; the join of
   environments has a function of that name too. *)
Lemma traversal_n_way_join_env_extension_renamed (env : state)
    (exts : list (Index.t * env_extension)) :
  FlambdaJoinTypes.n_way_join_env_extension join_env_extension'
    (rename_state env) (Keyed.map_values (rename_env_extension r) exts)
  = option_map (with_state (rename_env_extension r))
      (FlambdaJoinTypes.n_way_join_env_extension join_env_extension env exts).
Proof.
  unfold FlambdaJoinTypes.n_way_join_env_extension.
  rewrite join_env_extension_related.
  destruct (join_env_extension env exts) as [[[ext env'] | ] | ];
    reflexivity.
Qed.

Lemma generic_merge_map_known_renamed {K value value' other other' : Type}
    (eqb : K -> K -> bool) (gv : value -> value') (go : other -> other')
    (of_other : other -> value) (of_other' : other' -> value')
    (n_way_join :
       state -> list (Index.t * value)
       -> option (Or_bottom.t (value * state)))
    (n_way_join' :
       state -> list (Index.t * value')
       -> option (Or_bottom.t (value' * state))) :
  (∀ known : other, of_other' (go known) = gv (of_other known)) ->
  (∀ (env : state) (values : list (Index.t * value)),
    n_way_join' (rename_state env) (Keyed.map_values gv values)
    = option_map (or_bottom_map (with_state gv)) (n_way_join env values)) ->
  ∀ (env : state)
    (maps :
       list (Index.t * (Keyed.finite_map K value * Or_bottom.t other))),
    generic_merge_map_known eqb of_other' n_way_join' (rename_state env)
      (Keyed.map_values
         (fun map_and_other : Keyed.finite_map K value * Or_bottom.t other =>
            (Keyed.map_values gv (fst map_and_other),
             or_bottom_map go (snd map_and_other)))
         maps)
    = option_map (with_state (Keyed.map_values gv))
        (generic_merge_map_known eqb of_other n_way_join env maps).
Proof.
  intros Hof Hjoin env maps. unfold generic_merge_map_known.
  rewrite (gather_known_renamed eqb gv go of_other of_other' Hof).
  destruct maps as [ | input maps]; [ reflexivity | ].
  cbn [Keyed.map_values map].
  apply
    (keyed_filter_map_fold_values eqb (Keyed.map_values gv) gv rename_state).
  intros key values env'. rewrite Hjoin.
  destruct (n_way_join env' values) as [[[joined env''] | ] | ]; reflexivity.
Qed.

(* [n_way_join_row_like], with what it is given by its two callers and
   the action of the renaming on what a case maps to and on the known
   cases. The index of a case holds no variable. *)
Section Row_like_renamed.

Context {lattice shape maps_to known : Type}
  (rename_maps_to : maps_to -> maps_to) (rename_known : known -> known)
  (n_way_join_maps_to n_way_join_maps_to' :
     state -> shape -> list (Index.t * maps_to) -> option (maps_to * state))
  (equal_index : lattice -> lattice -> bool)
  (inter_index : lattice -> lattice -> lattice)
  (n_way_join_shape : list shape -> or_unknown_or_bottom shape)
  (merge_map_known merge_map_known' :
     (state
      -> list
           (Index.t
            * Or_unknown.t (row_like_case lattice shape maps_to env_extension))
      -> option
           (Or_bottom.t
              (Or_unknown.t (row_like_case lattice shape maps_to env_extension)
               * state)))
     -> state
     -> list
          (Index.t
           * (known
              * Or_bottom.t
                  (row_like_case lattice shape maps_to env_extension)))
     -> option (known * state)).

(* The action on a case: what it maps to and its extension. *)
Definition rename_case
    (case : row_like_case lattice shape maps_to env_extension)
    : row_like_case lattice shape maps_to env_extension :=
  Row_like_case (rename_maps_to (case_maps_to case)) (case_index case)
    (rename_env_extension r (case_env_extension case)).

Hypothesis n_way_join_maps_to_related :
  ∀ (env : state) (shape : shape) (maps_tos : list (Index.t * maps_to)),
    n_way_join_maps_to' (rename_state env) shape
      (Keyed.map_values rename_maps_to maps_tos)
    = option_map (with_state rename_maps_to)
        (n_way_join_maps_to env shape maps_tos).

Hypothesis merge_map_known_related :
  ∀ n_way_join_knowns n_way_join_knowns',
    (∀ (env : state)
       (cases :
          list
            (Index.t
             * Or_unknown.t
                 (row_like_case lattice shape maps_to env_extension))),
      n_way_join_knowns' (rename_state env)
        (Keyed.map_values (or_unknown_map rename_case) cases)
      = option_map (or_bottom_map (with_state (or_unknown_map rename_case)))
          (n_way_join_knowns env cases)) ->
    ∀ (env : state)
      (known_cases :
         list
           (Index.t
            * (known
               * Or_bottom.t
                   (row_like_case lattice shape maps_to env_extension)))),
      merge_map_known' n_way_join_knowns' (rename_state env)
        (Keyed.map_values
           (fun known_and_other =>
              (rename_known (fst known_and_other),
               or_bottom_map rename_case (snd known_and_other)))
           known_cases)
      = option_map (with_state rename_known)
          (merge_map_known n_way_join_knowns env known_cases).

Lemma n_way_join_case_renamed (env : state)
    (cases :
       list (Index.t * row_like_case lattice shape maps_to env_extension)) :
  n_way_join_case join_env_extension' n_way_join_maps_to' equal_index
    inter_index n_way_join_shape (rename_state env)
    (Keyed.map_values rename_case cases)
  = option_map (or_bottom_map (with_state (or_unknown_map rename_case)))
      (n_way_join_case join_env_extension n_way_join_maps_to equal_index
         inter_index n_way_join_shape env cases).
Proof.
  unfold n_way_join_case.
  replace
    (map (fun case => case_index (snd case))
       (Keyed.map_values rename_case cases))
    with (map (fun case => case_index (snd case)) cases)
    by (unfold Keyed.map_values; rewrite map_map; reflexivity).
  replace
    (map (fun case => (fst case, case_maps_to (snd case)))
       (Keyed.map_values rename_case cases))
    with
      (Keyed.map_values rename_maps_to
         (map (fun case => (fst case, case_maps_to (snd case))) cases))
    by (unfold Keyed.map_values; rewrite !map_map; reflexivity).
  replace
    (map (fun case => (fst case, case_env_extension (snd case)))
       (Keyed.map_values rename_case cases))
    with
      (Keyed.map_values (rename_env_extension r)
         (map (fun case => (fst case, case_env_extension (snd case))) cases))
    by (unfold Keyed.map_values; rewrite !map_map; reflexivity).
  destruct
    (n_way_join_index equal_index inter_index n_way_join_shape
       (map (fun case => case_index (snd case)) cases))
    as [ | index | ];
    [ reflexivity | | reflexivity ].
  rewrite n_way_join_maps_to_related.
  destruct
    (n_way_join_maps_to env (index_shape index)
       (map (fun case => (fst case, case_maps_to (snd case))) cases))
    as [[joined_maps_to env1] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite traversal_n_way_join_env_extension_renamed.
  destruct
    (FlambdaJoinTypes.n_way_join_env_extension join_env_extension env1
       (map (fun case => (fst case, case_env_extension (snd case))) cases))
    as [[joined_extension env2] | ];
    reflexivity.
Qed.

Lemma n_way_join_knowns_renamed (env : state)
    (cases :
       list
         (Index.t
          * Or_unknown.t (row_like_case lattice shape maps_to env_extension)))
    :
  n_way_join_knowns join_env_extension' n_way_join_maps_to' equal_index
    inter_index n_way_join_shape (rename_state env)
    (Keyed.map_values (or_unknown_map rename_case) cases)
  = option_map (or_bottom_map (with_state (or_unknown_map rename_case)))
      (n_way_join_knowns join_env_extension n_way_join_maps_to equal_index
         inter_index n_way_join_shape env cases).
Proof.
  unfold n_way_join_knowns. rewrite all_known_renamed.
  destruct (all_known cases) as [known_cases | ]; cbn [option_map];
    [ apply n_way_join_case_renamed | reflexivity ].
Qed.

Lemma n_way_join_row_like_renamed (env : state)
    (known_cases :
       list
         (Index.t
          * (known
             * Or_bottom.t
                 (row_like_case lattice shape maps_to env_extension))))
    (other :
       list
         (Index.t
          * Or_bottom.t (row_like_case lattice shape maps_to env_extension)))
    :
  n_way_join_row_like join_env_extension' n_way_join_maps_to' equal_index
    inter_index n_way_join_shape merge_map_known' (rename_state env)
    (Keyed.map_values
       (fun known_and_other =>
          (rename_known (fst known_and_other),
           or_bottom_map rename_case (snd known_and_other)))
       known_cases)
    (Keyed.map_values (or_bottom_map rename_case) other)
  = option_map
      (with_state
         (or_unknown_map
            (fun joined =>
               (rename_known (fst joined),
                or_bottom_map rename_case (snd joined)))))
      (n_way_join_row_like join_env_extension n_way_join_maps_to equal_index
         inter_index n_way_join_shape merge_map_known env known_cases other).
Proof.
  unfold n_way_join_row_like.
  rewrite (merge_map_known_related _ _ n_way_join_knowns_renamed).
  destruct
    (merge_map_known
       (n_way_join_knowns join_env_extension n_way_join_maps_to equal_index
          inter_index n_way_join_shape)
       env known_cases)
    as [[joined_known env1] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite
    (non_bottom_renamed rename_case [] other
     : non_bottom [] (Keyed.map_values (or_bottom_map rename_case) other)
       = _),
    n_way_join_case_renamed.
  destruct
    (n_way_join_case join_env_extension n_way_join_maps_to equal_index
       inter_index n_way_join_shape env1 (non_bottom [] other))
    as [[[[case | ] env2] | ] | ];
    reflexivity.
Qed.

End Row_like_renamed.

(* What is said of the join of types by the functions that call it:
   the two joins are supposed to be related. *)
Section Body_renamed.

Variables n_way_join_rec n_way_join_rec' :
  state -> list (Index.t * flambda_type)
  -> option (Or_unknown.t flambda_type * state).

Hypothesis n_way_join_rec_related :
  ∀ (env : state) (ts : list (Index.t * flambda_type)),
    n_way_join_rec' (rename_state env) (Keyed.map_values (rename_type r) ts)
    = option_map (with_state (or_unknown_map (rename_type r)))
        (n_way_join_rec env ts).

Lemma n_way_join_generic_product_renamed (env : state)
    (indexed_products : list (Index.t * finite_map flambda_type)) :
  n_way_join_generic_product n_way_join_rec' (rename_state env)
    (Keyed.map_values (map_values (rename_type r)) indexed_products)
  = option_map (with_state (map_values (rename_type r)))
      (n_way_join_generic_product n_way_join_rec env indexed_products).
Proof.
  unfold n_way_join_generic_product. rewrite gather_components_renamed.
  destruct indexed_products as [ | indexed_product indexed_products];
    [ reflexivity | ].
  cbn [Keyed.map_values map].
  apply
    (keyed_filter_map_fold_values Pos.eqb (Keyed.map_values (rename_type r))
       (rename_type r) rename_state).
  intros index tys env'. rewrite n_way_join_rec_related.
  destruct (n_way_join_rec env' tys) as [[[ty | ] env''] | ]; reflexivity.
Qed.

Lemma n_way_join_function_slot_indexed_product_renamed (env : state)
    (function_slots_indexed : list (Index.t * finite_map flambda_type)) :
  n_way_join_function_slot_indexed_product n_way_join_rec' (rename_state env)
    (Keyed.map_values (map_values (rename_type r)) function_slots_indexed)
  = option_map (with_state (map_values (rename_type r)))
      (n_way_join_function_slot_indexed_product n_way_join_rec env
         function_slots_indexed).
Proof.
  unfold n_way_join_function_slot_indexed_product.
  rewrite n_way_join_generic_product_renamed.
  destruct
    (n_way_join_generic_product n_way_join_rec env function_slots_indexed)
    as [[components env1] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite
    (for_all_bindings_b_values (rename_type r)
       (fun _ ty => kind_eqb (kind ty) Value)
       (fun _ ty => kind_eqb (kind ty) Value) components
       (fun _ ty =>
          f_equal (fun k => kind_eqb k Value) (kind_rename_type r ty))).
  destruct
    (for_all_bindings_b (fun _ ty => kind_eqb (kind ty) Value) components);
    reflexivity.
Qed.

Lemma n_way_join_value_slot_indexed_product_renamed (env : state)
    (value_slots_indexed : list (Index.t * finite_map flambda_type)) :
  n_way_join_value_slot_indexed_product n_way_join_rec' (rename_state env)
    (Keyed.map_values (map_values (rename_type r)) value_slots_indexed)
  = option_map (with_state (map_values (rename_type r)))
      (n_way_join_value_slot_indexed_product n_way_join_rec env
         value_slots_indexed).
Proof. apply n_way_join_generic_product_renamed. Qed.

Lemma fields_the_same_renamed (length : nat)
    (first_fields other_fields : list flambda_type) :
  fields_the_same type_eqb' length (map (rename_type r) first_fields)
    (map (rename_type r) other_fields)
  = fields_the_same type_eqb length first_fields other_fields.
Proof.
  revert first_fields other_fields.
  induction length as [ | length IH];
    intros [ | first_field first_fields] [ | other_field other_fields];
    try reflexivity.
  cbn [map fields_the_same]. rewrite type_eqb_related, IH. reflexivity.
Qed.

Lemma import_types_renamed (env : state) (tys : list flambda_type) :
  import_types import_type' (rename_state env) (map (rename_type r) tys)
  = option_map rename_state (import_types import_type env tys).
Proof.
  revert env. induction tys as [ | ty tys IH]; intros env; [ reflexivity | ].
  cbn [map import_types]. rewrite import_type_related.
  destruct (import_type env ty) as [env1 | ]; cbn [option_map];
    [ apply IH | reflexivity ].
Qed.

Lemma join_fields_renamed (env : state) (shape : block_shape)
    (fields : list (Index.t * list flambda_type)) (index count : nat) :
  join_fields n_way_join_rec' (rename_state env) shape
    (Keyed.map_values (map (rename_type r)) fields) index count
  = option_map (with_state (map (rename_type r)))
      (join_fields n_way_join_rec env shape fields index count).
Proof.
  revert env index.
  induction count as [ | count IH]; intros env index; [ reflexivity | ].
  cbn [join_fields]. rewrite fields_at_renamed.
  destruct (fields_at index fields) as [tys | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite n_way_join_rec_related.
  destruct (n_way_join_rec env tys) as [[joined env1] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  destruct joined as [ty | ]; cbn [or_unknown_map].
  - rewrite IH.
    destruct (join_fields n_way_join_rec env1 shape fields (S index) count)
      as [[tys' env2] | ];
      reflexivity.
  - unfold FlambdaJoinTypes.unknown_from_shape.
    destruct (block_shape_element_kind shape index) as [k | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite IH.
    destruct (join_fields n_way_join_rec env1 shape fields (S index) count)
      as [[tys' env2] | ];
      cbn [option_map fst snd map]; [ | reflexivity ].
    rewrite rename_type_unknown. reflexivity.
Qed.

Lemma fold_least_length_renamed {A B : Type} (g : A -> B)
    (others : list (Index.t * list A)) (least_length : nat) :
  List.fold_left
    (fun least_length (other_fields : Index.t * list B) =>
       Nat.min least_length (length (snd other_fields)))
    (Keyed.map_values (map g) others) least_length
  = List.fold_left
      (fun least_length (other_fields : Index.t * list A) =>
         Nat.min least_length (length (snd other_fields)))
      others least_length.
Proof.
  revert least_length.
  induction others as [ | [id fields] others IH]; intros least_length;
    [ reflexivity | ].
  rewrite keyed_map_values_cons. cbn [List.fold_left snd]. rewrite length_map.
  apply IH.
Qed.

Lemma n_way_join_int_indexed_product_renamed (env : state)
    (shape : block_shape) (fields : list (Index.t * list flambda_type)) :
  n_way_join_int_indexed_product import_type' type_eqb' n_way_join_rec'
    (rename_state env) shape
    (Keyed.map_values (map (rename_type r)) fields)
  = option_map (with_state (map (rename_type r)))
      (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec env
         shape fields).
Proof.
  destruct fields as [ | [first_id first_fields] other_fields];
    [ reflexivity | ].
  unfold n_way_join_int_indexed_product. rewrite keyed_map_values_cons.
  cbv zeta.
  rewrite fold_least_length_renamed, length_map.
  set
    (least_length :=
       List.fold_left
         (fun least_length (other_fields : Index.t * list flambda_type) =>
            Nat.min least_length (length (snd other_fields)))
         other_fields (length first_fields)).
  rewrite
    (keyed_forallb_values (map (rename_type r))
       (fun other_fields =>
          fields_the_same type_eqb least_length first_fields
            (snd other_fields))
       (fun other_fields =>
          fields_the_same type_eqb' least_length
            (map (rename_type r) first_fields) (snd other_fields))
       other_fields
       (fun other_fields =>
          fields_the_same_renamed least_length first_fields
            (snd other_fields))).
  destruct
    (forallb
       (fun other_fields =>
          fields_the_same type_eqb least_length first_fields
            (snd other_fields))
       other_fields).
  - rewrite <- keyed_map_values_cons.
    rewrite
      (keyed_find_values (map (rename_type r))
         (fun fields_of_id => Nat.eqb (length (snd fields_of_id)) least_length)
         (fun fields_of_id => Nat.eqb (length (snd fields_of_id)) least_length)
         ((first_id, first_fields) :: other_fields)
         (fun fields_of_id =>
            f_equal (fun n => Nat.eqb n least_length)
              (length_map (rename_type r) (snd fields_of_id)))).
    destruct
      (find
         (fun fields_of_id => Nat.eqb (length (snd fields_of_id)) least_length)
         ((first_id, first_fields) :: other_fields))
      as [[id fields_of_id] | ];
      cbn [option_map fst snd]; [ | reflexivity ].
    rewrite import_types_renamed.
    destruct (import_types import_type env fields_of_id); reflexivity.
  - rewrite <- keyed_map_values_cons. apply join_fields_renamed.
Qed.

Lemma n_way_join_closures_entry_renamed (env : state)
    (entries : list (Index.t * closures_entry)) :
  n_way_join_closures_entry n_way_join_rec' (rename_state env)
    (Keyed.map_values (rename_closures_entry r) entries)
  = option_map (with_state (rename_closures_entry r))
      (n_way_join_closures_entry n_way_join_rec env entries).
Proof.
  unfold n_way_join_closures_entry. cbv zeta.
  replace
    (map
       (fun entry => (fst entry, closures_entry_function_types (snd entry)))
       (Keyed.map_values (rename_closures_entry r) entries))
    with
      (map
         (fun entry => (fst entry, closures_entry_function_types (snd entry)))
         entries)
    by (unfold Keyed.map_values; rewrite map_map; apply map_ext;
        intros entry; cbn [fst snd];
        rewrite closures_entry_function_types_renamed; reflexivity).
  replace
    (map (fun entry => (fst entry, closures_entry_closure_types (snd entry)))
       (Keyed.map_values (rename_closures_entry r) entries))
    with
      (Keyed.map_values (map_values (rename_type r))
         (map
            (fun entry =>
               (fst entry, closures_entry_closure_types (snd entry)))
            entries))
    by (unfold Keyed.map_values; rewrite !map_map; apply map_ext;
        intros entry; cbn [fst snd];
        rewrite closures_entry_closure_types_renamed; reflexivity).
  replace
    (map
       (fun entry => (fst entry, closures_entry_value_slot_types (snd entry)))
       (Keyed.map_values (rename_closures_entry r) entries))
    with
      (Keyed.map_values (map_values (rename_type r))
         (map
            (fun entry =>
               (fst entry, closures_entry_value_slot_types (snd entry)))
            entries))
    by (unfold Keyed.map_values; rewrite !map_map; apply map_ext;
        intros entry; cbn [fst snd];
        rewrite closures_entry_value_slot_types_renamed; reflexivity).
  rewrite !keyed_map_values_rev,
    n_way_join_function_slot_indexed_product_renamed.
  destruct
    (n_way_join_function_slot_indexed_product n_way_join_rec env
       (rev
          (map
             (fun entry =>
                (fst entry, closures_entry_closure_types (snd entry)))
             entries)))
    as [[closure_types env1] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite n_way_join_value_slot_indexed_product_renamed.
  destruct
    (n_way_join_value_slot_indexed_product n_way_join_rec env1
       (rev
          (map
             (fun entry =>
                (fst entry, closures_entry_value_slot_types (snd entry)))
             entries)))
    as [[value_slot_types env2] | ];
    reflexivity.
Qed.

(* The action on a case of blocks is [rename_case] for the list of the
   types of its fields. *)
Local Abbreviation rename_block_case_r :=
  (rename_block_case (rename_type r) (rename_env_extension r)).

Lemma n_way_join_row_like_for_blocks_renamed (env : state)
    (row_likes : list (Index.t * row_like_for_blocks)) :
  n_way_join_row_like_for_blocks import_type' join_env_extension' type_eqb'
    block_shape_eqb n_way_join_rec' (rename_state env)
    (Keyed.map_values (rename_row_like_for_blocks r) row_likes)
  = option_map (with_state (or_unknown_map (rename_row_like_for_blocks r)))
      (n_way_join_row_like_for_blocks import_type join_env_extension type_eqb
         block_shape_eqb n_way_join_rec env row_likes).
Proof.
  unfold n_way_join_row_like_for_blocks. cbv zeta.
  pose
    (g_known :=
       fun known_and_other :
             Keyed.finite_map Z (Or_unknown.t row_like_block_case)
             * Or_bottom.t row_like_block_case =>
         (Keyed.map_values (or_unknown_map rename_block_case_r)
            (fst known_and_other),
          or_bottom_map rename_block_case_r (snd known_and_other))).
  pose
    (g_split :=
       or_bottom_map
         (fun split :
                list
                  (Index.t
                   * (Keyed.finite_map Z (Or_unknown.t row_like_block_case)
                      * Or_bottom.t row_like_block_case))
                * list (Index.t * Or_bottom.t row_like_block_case)
                * Alloc_mode.t =>
            (Keyed.map_values g_known (fst (fst split)),
             Keyed.map_values (or_bottom_map rename_block_case_r)
               (snd (fst split)),
             snd split))).
  match goal with
  | |- match List.fold_left ?f' _ _ with _ => _ end
       = option_map _ match List.fold_left ?f _ _ with _ => _ end =>
    assert
      (Hsplit :
         List.fold_left f'
           (Keyed.map_values (rename_row_like_for_blocks r) row_likes)
           Or_bottom.Bottom
         = g_split (List.fold_left f row_likes Or_bottom.Bottom));
      [ apply
          (fold_left_renamed g_split
             (fun row_like : Index.t * row_like_for_blocks =>
                (fst row_like, rename_row_like_for_blocks r (snd row_like)))
             f f' row_likes)
          with (init := Or_bottom.Bottom)
      | rewrite Hsplit; clear Hsplit;
        destruct (List.fold_left f row_likes Or_bottom.Bottom)
          as [[[known other] alloc_mode] | ] ]
  end.
  - intros acc row_like. cbn [fst snd].
    rewrite row_like_for_blocks_is_bottom_renamed.
    destruct (row_like_for_blocks_is_bottom (snd row_like));
      [ reflexivity | ].
    rewrite row_like_known_tags_renamed, row_like_other_tags_renamed,
      row_like_alloc_mode_renamed.
    destruct acc as [[[known other] alloc_mode] | ]; reflexivity.
  - cbn [g_split or_bottom_map fst snd].
    pose proof
      (n_way_join_row_like_renamed (map (rename_type r))
         (Keyed.map_values (or_unknown_map rename_block_case_r))
         (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec)
         (n_way_join_int_indexed_product import_type' type_eqb'
            n_way_join_rec')
         Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb)
         (generic_merge_map_known Z.eqb Or_unknown.Known)
         (generic_merge_map_known Z.eqb Or_unknown.Known)
         n_way_join_int_indexed_product_renamed
         (fun n_way_join_knowns n_way_join_knowns' Hknowns =>
            generic_merge_map_known_renamed Z.eqb
              (or_unknown_map rename_block_case_r) rename_block_case_r
              Or_unknown.Known Or_unknown.Known n_way_join_knowns
              n_way_join_knowns' (fun _ => eq_refl) Hknowns)
         env known other)
      as Hrow.
    match goal with
    | |- map_join_result _ ?join = _ => replace_by_equation join Hrow
    end.
    clear Hrow.
    apply
      (map_join_result_renamed
         (fun joined :
                Keyed.finite_map Z (Or_unknown.t row_like_block_case)
                * Or_bottom.t row_like_block_case =>
            (Keyed.map_values (or_unknown_map rename_block_case_r)
               (fst joined),
             or_bottom_map rename_block_case_r (snd joined)))
         (rename_row_like_for_blocks r)).
    intros joined. reflexivity.
  - reflexivity.
Qed.

(* The action on a case of closures is [rename_case] for the action on
   its entry. *)
Local Abbreviation rename_closures_case_r :=
  (rename_closures_case (rename_closures_entry r) (rename_env_extension r)).

Lemma n_way_join_row_like_for_closures_renamed (env : state)
    (closures : list (Index.t * row_like_for_closures)) :
  n_way_join_row_like_for_closures join_env_extension' n_way_join_rec'
    (rename_state env)
    (Keyed.map_values (rename_row_like_for_closures r) closures)
  = option_map (with_state (rename_row_like_for_closures r))
      (n_way_join_row_like_for_closures join_env_extension n_way_join_rec env
         closures).
Proof.
  unfold n_way_join_row_like_for_closures. cbv zeta.
  pose
    (g_known :=
       fun known_and_other :
             finite_map row_like_closures_case
             * Or_bottom.t row_like_closures_case =>
         (Keyed.map_values rename_closures_case_r (fst known_and_other),
          or_bottom_map rename_closures_case_r (snd known_and_other))).
  replace
    (rev
       (map
          (fun closures0 : Index.t * row_like_for_closures =>
             (fst closures0,
              (row_like_known_closures (snd closures0),
               @Or_bottom.Bottom row_like_closures_case)))
          (Keyed.map_values (rename_row_like_for_closures r) closures)))
    with
      (Keyed.map_values g_known
         (rev
            (map
               (fun closures0 : Index.t * row_like_for_closures =>
                  (fst closures0,
                   (row_like_known_closures (snd closures0),
                    @Or_bottom.Bottom row_like_closures_case)))
               closures)))
    by (rewrite <- keyed_map_values_rev; f_equal;
        unfold Keyed.map_values; rewrite !map_map; apply map_ext;
        intros closures0; unfold g_known; cbn [fst snd or_bottom_map];
        rewrite row_like_known_closures_renamed; reflexivity).
  match goal with
  | |- match n_way_join_row_like _ ?maps_to' _ _ ?shape ?merge' _ _ _
       with _ => _ end
       = option_map _
           match n_way_join_row_like _ ?maps_to _ _ _ ?merge _ ?known _
           with _ => _ end =>
    assert
      (Hmerge :
         ∀ n_way_join_knowns n_way_join_knowns',
           (∀ (env : state)
              (cases :
                 list (Index.t * Or_unknown.t row_like_closures_case)),
             n_way_join_knowns' (rename_state env)
               (Keyed.map_values (or_unknown_map rename_closures_case_r)
                  cases)
             = option_map
                 (or_bottom_map
                    (with_state (or_unknown_map rename_closures_case_r)))
                 (n_way_join_knowns env cases)) ->
           ∀ (env : state)
             (known_cases :
                list
                  (Index.t
                   * (finite_map row_like_closures_case
                      * Or_bottom.t row_like_closures_case))),
             merge' n_way_join_knowns' (rename_state env)
               (Keyed.map_values g_known known_cases)
             = option_map
                 (with_state (Keyed.map_values rename_closures_case_r))
                 (merge n_way_join_knowns env known_cases));
      [ | pose proof
            (n_way_join_row_like_renamed (rename_closures_entry r)
               (Keyed.map_values rename_closures_case_r) maps_to maps_to'
               set_of_closures_contents_equal set_of_closures_contents_inter
               shape merge merge'
               (fun env (_ : unit) entries =>
                  n_way_join_closures_entry_renamed env entries)
               Hmerge env known [])
            as Hrow ]
  end.
  - intros n_way_join_knowns n_way_join_knowns' Hknowns env0 known_cases.
    cbv beta.
    apply
      (generic_merge_map_known_renamed Pos.eqb rename_closures_case_r
         rename_closures_case_r (fun case => case) (fun case => case));
      [ reflexivity | ].
    intros env1 values. cbv beta zeta.
    replace
      (map
         (fun case : Index.t * row_like_closures_case =>
            (fst case, Or_unknown.Known (snd case)))
         (Keyed.map_values rename_closures_case_r values))
      with
        (Keyed.map_values (or_unknown_map rename_closures_case_r)
           (map
              (fun case : Index.t * row_like_closures_case =>
                 (fst case, Or_unknown.Known (snd case)))
              values))
      by (unfold Keyed.map_values; rewrite !map_map; reflexivity).
    rewrite Hknowns.
    match goal with
    | |- _ = option_map _ match ?join with _ => _ end =>
      destruct join as [[[[case | ] env2] | ] | ]; reflexivity
    end.
  - match goal with
    | |- match ?join with _ => _ end = _ => replace_by_equation join Hrow
    end.
    clear Hrow Hmerge.
    match goal with
    | |- _ = option_map _ match ?join with _ => _ end =>
      destruct join as [[[[known_closures [case | ]] | ] env1] | ];
        reflexivity
    end.
Qed.

Lemma n_way_join_variant_renamed (env : state)
    (blocks : list (Index.t * Or_unknown.t row_like_for_blocks))
    (imms : list (Index.t * Or_unknown.t flambda_type))
    (extensions : list (Index.t * variant_extensions)) :
  n_way_join_variant import_type' join_env_extension' type_eqb'
    block_shape_eqb n_way_join_rec' (rename_state env)
    (Keyed.map_values (or_unknown_map (rename_row_like_for_blocks r)) blocks)
    (Keyed.map_values (or_unknown_map (rename_type r)) imms)
    (Keyed.map_values (rename_variant_extensions r) extensions)
  = option_map
      (with_state
         (or_unknown_map
            (fun joined :
                   Or_unknown.t row_like_for_blocks
                   * Or_unknown.t flambda_type * variant_extensions =>
               (or_unknown_map (rename_row_like_for_blocks r)
                  (fst (fst joined)),
                or_unknown_map (rename_type r) (snd (fst joined)),
                rename_variant_extensions r (snd joined)))))
      (n_way_join_variant import_type join_env_extension type_eqb
         block_shape_eqb n_way_join_rec env blocks imms extensions).
Proof.
  unfold n_way_join_variant.
  rewrite
    (n_way_join_unknown_renamed (rename_row_like_for_blocks r)
       (rename_row_like_for_blocks r) _ _
       n_way_join_row_like_for_blocks_renamed).
  match goal with
  | |- _ = option_map _ match ?join with _ => _ end =>
    destruct join as [[joined_blocks env1] | ]
  end;
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite
    (n_way_join_unknown_renamed (rename_type r) (rename_type r) _ _
       n_way_join_rec_related).
  destruct (n_way_join_unknown n_way_join_rec env1 imms)
    as [[joined_imms env2] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite
    (gather_extensions_renamed r [] [] extensions
     : gather_extensions [] []
         (Keyed.map_values (rename_variant_extensions r) extensions)
       = _).
  destruct (gather_extensions [] [] extensions)
    as [[when_immediates when_blocks] | ];
    cbn [option_map fst snd].
  - rewrite traversal_n_way_join_env_extension_renamed.
    destruct
      (FlambdaJoinTypes.n_way_join_env_extension join_env_extension env2
         when_immediates)
      as [[when_immediate env3] | ];
      cbn [option_map fst snd]; [ | reflexivity ].
    rewrite traversal_n_way_join_env_extension_renamed.
    destruct
      (FlambdaJoinTypes.n_way_join_env_extension join_env_extension env3
         when_blocks)
      as [[when_block env4] | ];
      cbn [option_map fst snd]; [ | reflexivity ].
    rewrite !env_extension_is_empty_renamed.
    destruct
      (env_extension_is_empty when_immediate
       && env_extension_is_empty when_block);
      destruct joined_blocks, joined_imms; reflexivity.
  - destruct joined_blocks, joined_imms; reflexivity.
Qed.

Lemma n_way_join_head_of_kind_value_non_null_renamed (env : state)
    (heads : list (Index.t * head_of_kind_value_non_null)) :
  n_way_join_head_of_kind_value_non_null n_way_join_simples' import_type'
    join_env_extension' type_eqb' block_shape_eqb n_way_join_rec'
    (rename_state env)
    (Keyed.map_values (rename_head_of_kind_value_non_null r) heads)
  = option_map
      (with_state (or_unknown_map (rename_head_of_kind_value_non_null r)))
      (n_way_join_head_of_kind_value_non_null n_way_join_simples import_type
         join_env_extension type_eqb block_shape_eqb n_way_join_rec env
         heads).
Proof.
  destruct heads as [ | [first_id first_head] other_heads];
    [ reflexivity | ].
  rewrite keyed_map_values_cons.
  destruct first_head
    as [is_int immediates get_tag blocks extensions is_unique | alloc_mode
       | k n alloc_mode | by_function_slot alloc_mode | strs];
    cbn [rename_head_of_kind_value_non_null
           n_way_join_head_of_kind_value_non_null].
  - match goal with
    | |- context [first_variant_parts first_id ?first'] =>
      change first'
        with
          (rename_variant_fields r
             {| variant_is_int := is_int;
                variant_immediates := immediates;
                variant_get_tag := get_tag;
                variant_blocks := blocks;
                variant_extensions_of := extensions;
                variant_is_unique := is_unique |})
    end.
    rewrite first_variant_parts_renamed.
    rewrite
      (gather_heads_renamed r (rename_variant_fields r)
         (rename_variant_parts r) read_variant read_variant add_variant_parts
         add_variant_parts other_heads _ (read_variant_renamed r)
         (add_variant_parts_renamed r)).
    match goal with
    | |- _ = option_map _ match ?gathered with _ => _ end =>
      destruct gathered as [parts | ]
    end;
      cbn [option_map]; [ | reflexivity ].
    cbn [rename_variant_parts parts_blocks parts_immediates parts_extensions
           parts_is_unique parts_is_int parts_get_tag].
    rewrite n_way_join_relation_simples_renamed.
    destruct
      (n_way_join_relation_simples n_way_join_simples env
         (parts_is_int parts))
      as [[joined_is_int env1] | ];
      cbn [option_map fst snd]; [ | reflexivity ].
    rewrite n_way_join_relation_simples_renamed.
    destruct
      (n_way_join_relation_simples n_way_join_simples env1
         (parts_get_tag parts))
      as [[joined_get_tag env2] | ];
      cbn [option_map fst snd]; [ | reflexivity ].
    rewrite n_way_join_variant_renamed.
    apply
      (map_join_result_renamed
         (fun joined :
                Or_unknown.t row_like_for_blocks * Or_unknown.t flambda_type
                * variant_extensions =>
            (or_unknown_map (rename_row_like_for_blocks r) (fst (fst joined)),
             or_unknown_map (rename_type r) (snd (fst joined)),
             rename_variant_extensions r (snd joined)))
         (rename_head_of_kind_value_non_null r)).
    intros [[joined_blocks joined_imms] joined_extensions]. reflexivity.
  - rewrite
      (gather_heads_renamed r
         (fun other_alloc_mode : Alloc_mode.t => other_alloc_mode)
         (fun alloc_mode : Alloc_mode.t => alloc_mode) read_mutable_block
         read_mutable_block
         (fun _ other_alloc_mode alloc_mode =>
            join_alloc_mode alloc_mode other_alloc_mode)
         (fun _ other_alloc_mode alloc_mode =>
            join_alloc_mode alloc_mode other_alloc_mode)
         other_heads alloc_mode);
      [ | intros head; rewrite read_mutable_block_renamed;
          destruct (read_mutable_block head); reflexivity
        | reflexivity ].
    match goal with
    | |- _ = option_map _ match ?gathered with _ => _ end =>
      destruct gathered as [joined_alloc_mode | ]
    end;
      reflexivity.
  - pose
      (g_acc :=
         fun acc : list (Index.t * flambda_type) * Alloc_mode.t =>
           (Keyed.map_values (rename_type r) (fst acc), snd acc)).
    pose proof
      (gather_heads_renamed r
         (fun contents : flambda_type * Alloc_mode.t =>
            (rename_type r (fst contents), snd contents))
         g_acc (read_boxed_number k) (read_boxed_number k)
         (fun other_id other acc =>
            ((other_id, fst other) :: fst acc,
             join_alloc_mode (snd acc) (snd other)))
         (fun other_id other acc =>
            ((other_id, fst other) :: fst acc,
             join_alloc_mode (snd acc) (snd other)))
         other_heads
         ([(first_id, n)], alloc_mode) (read_boxed_number_renamed r k)
         (fun _ _ _ => eq_refl))
      as Hgather.
    match goal with
    | |- match ?gathered with _ => _ end = _ =>
      replace_by_equation gathered Hgather
    end.
    clear Hgather.
    match goal with
    | |- _ = option_map _ match ?gathered with _ => _ end =>
      destruct gathered as [[ns joined_alloc_mode] | ]
    end;
      cbn [option_map g_acc fst snd]; [ | reflexivity ].
    rewrite n_way_join_rec_related.
    apply
      (map_join_result_renamed (rename_type r)
         (rename_head_of_kind_value_non_null r)).
    reflexivity.
  - pose
      (g_acc :=
         fun acc : list (Index.t * row_like_for_closures) * Alloc_mode.t =>
           (Keyed.map_values (rename_row_like_for_closures r) (fst acc),
            snd acc)).
    pose proof
      (gather_heads_renamed r
         (fun closures : row_like_for_closures * Alloc_mode.t =>
            (rename_row_like_for_closures r (fst closures), snd closures))
         g_acc read_closures read_closures
         (fun other_id other acc =>
            ((other_id, fst other) :: fst acc,
             join_alloc_mode (snd acc) (snd other)))
         (fun other_id other acc =>
            ((other_id, fst other) :: fst acc,
             join_alloc_mode (snd acc) (snd other)))
         other_heads
         ([(first_id, by_function_slot)], alloc_mode)
         (read_closures_renamed r) (fun _ _ _ => eq_refl))
      as Hgather.
    match goal with
    | |- match ?gathered with _ => _ end = _ =>
      replace_by_equation gathered Hgather
    end.
    clear Hgather.
    match goal with
    | |- _ = option_map _ match ?gathered with _ => _ end =>
      destruct gathered as [[function_slots joined_alloc_mode] | ]
    end;
      cbn [option_map g_acc fst snd]; [ | reflexivity ].
    rewrite n_way_join_row_like_for_closures_renamed.
    destruct
      (n_way_join_row_like_for_closures join_env_extension n_way_join_rec env
         function_slots)
      as [[joined env1] | ];
      reflexivity.
  - rewrite
      (gather_heads_renamed r (fun other_strs : list (list Z) => other_strs)
         (fun all_strs : list (list Z) => all_strs) read_string read_string
         (fun _ other_strs all_strs => strings_union other_strs all_strs)
         (fun _ other_strs all_strs => strings_union other_strs all_strs)
         other_heads strs);
      [ | intros head; rewrite read_string_renamed;
          destruct (read_string head); reflexivity
        | reflexivity ].
    match goal with
    | |- _ = option_map _ match ?gathered with _ => _ end =>
      destruct gathered as [all_strs | ]
    end;
      reflexivity.
Qed.

Lemma n_way_join_head_of_kind_value_renamed (env : state)
    (heads : list (Index.t * head_of_kind_value)) :
  n_way_join_head_of_kind_value n_way_join_simples' import_type'
    join_env_extension' type_eqb' block_shape_eqb n_way_join_rec'
    (rename_state env)
    (Keyed.map_values (rename_head_of_kind_value r) heads)
  = option_map (with_state (or_unknown_map (rename_head_of_kind_value r)))
      (n_way_join_head_of_kind_value n_way_join_simples import_type
         join_env_extension type_eqb block_shape_eqb n_way_join_rec env
         heads).
Proof.
  unfold n_way_join_head_of_kind_value.
  rewrite n_way_join_is_null_renamed.
  destruct (n_way_join_is_null n_way_join_simples env heads)
    as [[is_null env1] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite
    (non_null_heads_renamed r [] heads
     : non_null_heads [] (Keyed.map_values (rename_head_of_kind_value r) heads)
       = _).
  destruct (non_null_heads [] heads) as [[ | non_null non_nulls] | ];
    cbn [option_map].
  - destruct is_null as [ | [var | ]]; reflexivity.
  - rewrite
      (n_way_join_head_of_kind_value_non_null_renamed env1
         (non_null :: non_nulls)).
    cbn [Keyed.map_values map].
    destruct
      (n_way_join_head_of_kind_value_non_null n_way_join_simples import_type
         join_env_extension type_eqb block_shape_eqb n_way_join_rec env1
         (non_null :: non_nulls))
      as [[[head | ] env2] | ];
      try reflexivity;
      destruct is_null as [ | [var | ]]; reflexivity.
  - destruct is_null as [ | [var | ]]; reflexivity.
Qed.

Lemma n_way_join_expanded_head_renamed (env : state) (kind : flambda_kind)
    (expandeds : list (Index.t * Expanded_type.t)) :
  n_way_join_expanded_head joined_env' n_way_join_simples' import_type'
    join_env_extension' reduce' type_eqb' block_shape_eqb n_way_join_rec'
    (rename_state env) kind
    (Keyed.map_values (rename_expanded r) expandeds)
  = option_map (with_state (or_unknown_map (rename_expanded r)))
      (n_way_join_expanded_head joined_env n_way_join_simples import_type
         join_env_extension reduce type_eqb block_shape_eqb n_way_join_rec
         env kind expandeds).
Proof.
  induction expandeds as [ | [id1 [kind1 [ | descr1 | ]]] expandeds IH];
    [ reflexivity | reflexivity | | exact IH ].
  rewrite keyed_map_values_cons.
  destruct descr1 as [head1 | head1 | k head1 | head1 | head1];
    cbn [n_way_join_expanded_head rename_expanded rename_expanded_head
           Expanded_type.descr Expanded_type.kind or_unknown_or_bottom_map].
  - match goal with
    | |- context [extract_heads ?extract (Keyed.map_values _ expandeds)] =>
      rewrite
        (extract_heads_renamed r (rename_head_of_kind_value r) extract extract
           expandeds)
        by (intros [head | head | k' head | head | head];
            cbn [rename_expanded_head option_map]; try reflexivity;
            destruct (kind_eqb _ _); reflexivity)
    end.
    match goal with
    | |- _ = option_map _ match match ?extracted with _ => _ end with
                          | _ => _
                          end =>
      destruct extracted as [[heads | ] | ]
    end;
      try reflexivity.
    cbn [option_map].
    rewrite <- keyed_map_values_cons, n_way_join_head_of_kind_value_renamed.
    match goal with
    | |- _ = option_map _ match map_join_result _ ?join with _ => _ end =>
      destruct join as [[[head | ] env1] | ]
    end;
      reflexivity.
  - match goal with
    | |- context [extract_heads ?extract (Keyed.map_values _ expandeds)] =>
      rewrite
        (extract_heads_renamed r (rename_head_of_kind_naked_immediate r)
           extract extract expandeds)
        by (intros [head | head | k' head | head | head];
            cbn [rename_expanded_head option_map]; try reflexivity;
            destruct (kind_eqb _ _); reflexivity)
    end.
    match goal with
    | |- _ = option_map _ match match ?extracted with _ => _ end with
                          | _ => _
                          end =>
      destruct extracted as [[heads | ] | ]
    end;
      try reflexivity.
    cbn [option_map].
    rewrite <- keyed_map_values_cons,
      n_way_join_head_of_kind_naked_immediate_renamed.
    match goal with
    | |- _ = option_map _ match map_join_result _ ?join with _ => _ end =>
      destruct join as [[[head | ] env1] | ]
    end;
      reflexivity.
  - match goal with
    | |- context [extract_heads ?extract (Keyed.map_values _ expandeds)] =>
      rewrite
        (extract_heads_renamed r (fun numbers : list Z => numbers) extract
           extract expandeds)
        by (intros [head | head | k' head | head | head];
            cbn [rename_expanded_head option_map]; try reflexivity;
            destruct (kind_eqb _ _); reflexivity)
    end.
    match goal with
    | |- _ = option_map _ match match ?extracted with _ => _ end with
                          | _ => _
                          end =>
      destruct extracted as [[heads | ] | ]
    end;
      try reflexivity.
    cbn [option_map]. rewrite keyed_map_values_fixes by reflexivity.
    reflexivity.
  - match goal with
    | |- context [extract_heads ?extract (Keyed.map_values _ expandeds)] =>
      rewrite
        (extract_heads_renamed r (fun head : unit => head) extract extract
           expandeds)
        by (intros [head | head | k' head | head | head];
            cbn [rename_expanded_head option_map]; try reflexivity;
            destruct (kind_eqb _ _); reflexivity)
    end.
    match goal with
    | |- _ = option_map _ match match ?extracted with _ => _ end with
                          | _ => _
                          end =>
      destruct extracted as [[heads | ] | ]
    end;
      reflexivity.
  - match goal with
    | |- context [extract_heads ?extract (Keyed.map_values _ expandeds)] =>
      rewrite
        (extract_heads_renamed r (fun head : unit => head) extract extract
           expandeds)
        by (intros [head | head | k' head | head | head];
            cbn [rename_expanded_head option_map]; try reflexivity;
            destruct (kind_eqb _ _); reflexivity)
    end.
    match goal with
    | |- _ = option_map _ match match ?extracted with _ => _ end with
                          | _ => _
                          end =>
      destruct extracted as [[heads | ] | ]
    end;
      reflexivity.
Qed.

Lemma expand_heads_renamed (env : state)
    (ts : list (Index.t * flambda_type)) :
  expand_heads joined_env' expand_head' (rename_state env)
    (Keyed.map_values (rename_type r) ts)
  = option_map (Keyed.map_values (rename_expanded r))
      (expand_heads joined_env expand_head env ts).
Proof.
  induction ts as [ | [id ty] ts IH]; [ reflexivity | ].
  rewrite keyed_map_values_cons. cbn [expand_heads].
  rewrite joined_env_related.
  destruct (joined_env env id) as [tenv | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite expand_head_related.
  destruct (expand_head tenv ty) as [expanded | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite IH.
  destruct (expand_heads joined_env expand_head env ts); reflexivity.
Qed.

Lemma n_way_join_body_renamed (env : state)
    (ts : list (Index.t * flambda_type)) :
  n_way_join_body joined_env' n_way_join_simples' import_type'
    join_env_extension' expand_head' reduce' type_eqb' block_shape_eqb
    n_way_join_rec' (rename_state env) (Keyed.map_values (rename_type r) ts)
  = option_map (with_state (or_unknown_map (rename_type r)))
      (n_way_join_body joined_env n_way_join_simples import_type
         join_env_extension expand_head reduce type_eqb block_shape_eqb
         n_way_join_rec env ts).
Proof.
  destruct ts as [ | [id1 t1] others]; [ reflexivity | ].
  unfold n_way_join_body. rewrite keyed_map_values_cons. cbv zeta.
  rewrite kind_rename_type.
  rewrite
    (keyed_forallb_values (rename_type r)
       (fun t => kind_eqb (kind t1) (kind (snd t)))
       (fun t => kind_eqb (kind t1) (kind (snd t))) others
       (fun t => f_equal (kind_eqb (kind t1)) (kind_rename_type r (snd t)))).
  destruct
    (negb (forallb (fun t => kind_eqb (kind t1) (kind (snd t))) others));
    [ reflexivity | ].
  rewrite <- keyed_map_values_cons.
  rewrite
    (keyed_filter_values (rename_type r)
       (fun t => negb (is_obviously_bottom (snd t)))
       (fun t => negb (is_obviously_bottom (snd t))) ((id1, t1) :: others)
       (fun t => f_equal negb (is_obviously_bottom_renamed r (snd t)))).
  rewrite alias_simples_renamed.
  destruct
    (alias_simples
       (filter (fun t => negb (is_obviously_bottom (snd t)))
          ((id1, t1) :: others)))
    as [canonical_simples | ];
    cbn [option_map].
  - rewrite n_way_join_simples_related.
    destruct (n_way_join_simples env (kind t1) canonical_simples)
      as [[[simple | ] env1] | ];
      cbn [option_map or_bottom_map fst snd or_unknown_map];
      rewrite ?rename_type_alias_type_of, ?rename_type_bottom; reflexivity.
  - rewrite expand_heads_renamed.
    destruct
      (expand_heads joined_env expand_head env
         (filter (fun t => negb (is_obviously_bottom (snd t)))
            ((id1, t1) :: others)))
      as [expandeds | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite n_way_join_expanded_head_renamed.
    apply (map_join_result_renamed (rename_expanded r) (rename_type r)).
    apply to_type_renamed.
Qed.

End Body_renamed.

(* [n_way_join]: by induction on the number that bounds its depth. *)
Lemma n_way_join_renamed (fuel : nat) (env : state)
    (ts : list (Index.t * flambda_type)) :
  n_way_join joined_env' n_way_join_simples' import_type' join_env_extension'
    expand_head' reduce' type_eqb' block_shape_eqb fuel (rename_state env)
    (Keyed.map_values (rename_type r) ts)
  = option_map
      (fun result =>
         (or_unknown_map (rename_type r) (fst result),
          rename_state (snd result)))
      (n_way_join joined_env n_way_join_simples import_type
         join_env_extension expand_head reduce type_eqb block_shape_eqb fuel
         env ts).
Proof.
  revert env ts. induction fuel as [ | fuel IH]; intros env ts;
    [ reflexivity | ].
  exact (n_way_join_body_renamed _ _ IH env ts).
Qed.

End Traversal_renamed.
