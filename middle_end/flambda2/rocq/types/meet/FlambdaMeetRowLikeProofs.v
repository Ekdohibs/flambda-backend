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


(* Proofs for FlambdaMeetRowLikeSpec.v. Nothing here needs review.

   The first part is about [meet_row_like] whatever the cases are of:
   [row_like_body_ok]. The function is unfolded into parts
   ([meet_row_like_unfold]). The cases are met one pair after the
   other, with a state: the environments of the pairs that stay, the
   variables that their levels define (the extra variables), and the
   names to avoid. What is kept of the state ([row_inv]): each
   environment kept is the environment of the meet one scope further,
   extended, and fits; two of them share only what the environment of
   the meet knows; the variables that are new in one are to be
   avoided; an extra variable is one that the level of an environment
   kept defines, with its kind. What is kept of a pair: the case
   answered fits the environment of the meet or an environment kept
   whose level the extra variables cover ([case_placed]); and where
   the two cases are true of the value in hand, an environment kept
   has a valuation, under which the case answered is true
   ([pair_sound]). At the end the environments kept are joined and
   the extra variables defined ([row_join_all_spec]): the environment
   answered knows what each environment kept knows, and the valuation
   of the pair of the tag of the value extends to it.

   The second part is about the rows of blocks:
   [meet_row_like_for_blocks_ok]. What the first part asks is given
   for blocks. The index of a case and its product are well formed
   together (conditions 6 and 7 of the well-formedness of types); the
   index is true of the block in hand, and the product of its fields;
   the index then says that the block has a field for each type of
   the product ([block_fields_hold]). Two indexes that a block both
   has are of one shape, and meet ([block_index_true_meet]); the index
   answered and the product answered are well formed together, the
   product answered being as long as the longer input
   ([block_index_ok_meet]). The merge of the maps goes through the
   tags of both, each once ([block_merge_map_known_spec]). A row
   fits, and is true, as the type that it makes
   ([blocks_type_fits_row], [block_row_fits_type],
   [block_row_true_satisfied]). The block in hand is the one that the
   value is, which the static environment says. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs
  FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaTypeEqualSpec
  FlambdaTypeEqualProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetStepsSpec
  FlambdaMeetStepsProofs FlambdaMeetEnvSpec FlambdaMeetEnvProofs
  FlambdaMeetScopesSpec FlambdaMeetScopesProofs FlambdaMeetProductsSpec
  FlambdaMeetProductsProofs FlambdaMeetHeadsSpec FlambdaMeetHeadsProofs
  FlambdaMeetRowLikeSpec.
Import ListNotations.

(* The proofs below, as far as the end of the section, are for any
   predicate on static environments that implies validity
   (FlambdaMeetValidity.v): in them the names of the four contracts
   that suppose a valid static environment stand for the same with
   that predicate. The theorems of the signed statements are after the
   section, at validity itself. *)
From OxCamlFlambda2Types Require Import FlambdaMeetValidity.

Section Validity.

Context {V : validity}.

Abbreviation statics_valid := valid.
Abbreviation meet_expanded_head_sound := meet_expanded_head_sound_for.
Abbreviation join_scoped_envs_sound := join_scoped_envs_sound_for.
Abbreviation meet_call_ok := meet_call_ok_for.
Abbreviation part_meet_ok := part_meet_ok_for.
Abbreviation meet_disjunction_right := meet_disjunction_right_for.
Abbreviation type_part_meet_ok := type_part_meet_ok_for.
Abbreviation meet_row_like_for_closures_right :=
  meet_row_like_for_closures_right_for.
Abbreviation value_heads_right := value_heads_right_for.

(* ======================================================================== *)
(* The extra variables                                                      *)
(* ======================================================================== *)

(* A list of variables with their kinds in which a variable has one
   kind: what is found for a variable is each of its bindings. *)
Definition one_kind (extra : list (variable * flambda_kind)) : Prop :=
  ∀ (x : variable) (k : flambda_kind),
    (x, k) ∈ extra -> lookup x extra = Some k.

Lemma one_kind_nil : one_kind [].
Proof. intros x k []. Qed.

Lemma lookup_app_some {A : Type} (x : positive) (a : A)
    (l l' : list (positive * A)) :
  lookup x l = Some a -> lookup x (l ++ l') = Some a.
Proof. intros H. rewrite lookup_app, H. reflexivity. Qed.

Lemma in_lookup_some {A : Type} (x : positive) (a : A)
    (l : list (positive * A)) :
  (x, a) ∈ l -> ∃ a' : A, lookup x l = Some a'.
Proof.
  intros Hin. destruct (lookup x l) as [ a' | ] eqn:Hlookup;
    [ exists a'; reflexivity | ].
  exfalso. apply (lookup_none_not_in x l Hlookup). apply in_map_iff.
  exists (x, a). split; [ reflexivity | exact Hin ].
Qed.

(* What [union_extra_variables] answers. *)
Lemma union_extra_variables_spec (vars : list (variable * flambda_kind)) :
  ∀ extra extra' : list (variable * flambda_kind),
    one_kind extra ->
    union_extra_variables extra vars = Some extra' ->
    one_kind extra'
    ∧ (∀ (y : variable) (k : flambda_kind),
         lookup y extra = Some k -> lookup y extra' = Some k)
    ∧ (∀ (y : variable) (k : flambda_kind),
         (y, k) ∈ vars -> lookup y extra' = Some k)
    ∧ (∀ (y : variable) (k : flambda_kind),
         lookup y extra' = Some k ->
         lookup y extra = Some k ∨ (y, k) ∈ vars).
Proof.
  induction vars as [ | [x kind] vars IH ]; intros extra extra' Hone Hunion.
  - injection Hunion as <-. split; [ exact Hone | ].
    split; [ auto | ]. split; [ intros y k [] | auto ].
  - cbn [union_extra_variables] in Hunion.
    destruct (lookup x extra) as [ kind' | ] eqn:Hx.
    + destruct (kind_eqb kind kind') eqn:Hkind; [ | discriminate Hunion ].
      apply kind_eqb_eq in Hkind. subst kind'.
      destruct (IH extra extra' Hone Hunion) as (Hone' & Hkept & Hvars & Hfrom).
      split; [ exact Hone' | ]. split; [ exact Hkept | ]. split.
      * intros y k [ [= <- <-] | Hin ]; [ exact (Hkept x kind Hx) | ].
        exact (Hvars y k Hin).
      * intros y k Hy. destruct (Hfrom y k Hy) as [ H | H ]; [ left | right ];
          [ exact H | right; exact H ].
    + assert (Hone1 : one_kind (extra ++ [(x, kind)])).
      { intros y k Hin. apply in_app_or in Hin as [ Hin | [ [= <- <-] | [] ] ].
        - exact (lookup_app_some y k extra _ (Hone y k Hin)).
        - rewrite lookup_app, Hx. cbn. rewrite Pos.eqb_refl. reflexivity. }
      destruct (IH (extra ++ [(x, kind)]) extra' Hone1 Hunion)
        as (Hone' & Hkept & Hvars & Hfrom).
      split; [ exact Hone' | ]. split.
      { intros y k Hy. apply Hkept. exact (lookup_app_some y k extra _ Hy). }
      split.
      * intros y k [ [= <- <-] | Hin ]; [ | exact (Hvars y k Hin) ].
        apply Hkept. rewrite lookup_app, Hx. cbn. rewrite Pos.eqb_refl.
        reflexivity.
      * intros y k Hy. destruct (Hfrom y k Hy) as [ H | H ];
          [ | right; right; exact H ].
        rewrite lookup_app in H. destruct (lookup y extra) as [ k' | ] eqn:Hy0.
        -- left. exact H.
        -- right. left. cbn in H.
           destruct (Pos.eqb y x) eqn:Hyx; [ | discriminate H ].
           apply Pos.eqb_eq in Hyx. subst y. injection H as <-. reflexivity.
Qed.

(* The variables of a list are defined, one after the other: as the
   variables that the list has, each with the kind that it has there,
   are. *)
Lemma define_extra_all (choices : meet_choices)
    (kind_of : variable -> option flambda_kind)
    (rest : list (variable * flambda_kind)) :
  (∀ (x : variable) (k : flambda_kind), (x, k) ∈ rest -> kind_of x = Some k) ->
  ∀ acc : option Meet_env.t,
    List.fold_left
      (fun env binding =>
         let* env := env in
         define_in_types choices env (fst binding) (snd binding))
      rest acc
    = define_all choices kind_of (map fst rest) acc.
Proof.
  induction rest as [ | [x k] rest IH ]; intros Hkinds acc; [ reflexivity | ].
  cbn [List.fold_left map fst snd define_all].
  rewrite (Hkinds x k (or_introl eq_refl)).
  apply IH. intros y k' Hin. exact (Hkinds y k' (or_intror Hin)).
Qed.

(* ======================================================================== *)
(* A merge of two maps that threads a state                                 *)
(* ======================================================================== *)

Lemma insert_number_permutation (i : Z) (sorted : list Z) :
  Permutation (insert_number i sorted) (i :: sorted).
Proof.
  induction sorted as [ | j rest IH ]; cbn; [ reflexivity | ].
  destruct (Z.leb i j); [ reflexivity | ].
  rewrite IH. apply perm_swap.
Qed.

Lemma sort_numbers_permutation (numbers : list Z) :
  Permutation (sort_numbers numbers) numbers.
Proof.
  induction numbers as [ | i numbers IH ]; cbn; [ reflexivity | ].
  rewrite insert_number_permutation. apply perm_skip. exact IH.
Qed.

Section Merge.

Context {K A B C S : Type}.

Variable eqb : K -> K -> bool.

Hypothesis Heqb : decides_equality eqb.

Variable f : K -> option A -> option B -> S -> option (option C * S).
Variable m1 : Keyed.finite_map K A.
Variable m2 : Keyed.finite_map K B.

(* A step of [merge_fold_failing]. *)
Definition failing_step (acc : Keyed.finite_map K C * option S) (key : K)
    : Keyed.finite_map K C * option S :=
  let '(c, s) :=
    match snd acc with
    | None => (None, None)
    | Some state =>
      match f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2) state
      with
      | None => (None, None)
      | Some (c, state) => (c, Some state)
      end
    end
  in
  (match c with
   | Some c => fst acc ++ [(key, c)]
   | None => fst acc
   end,
   s).

Lemma merge_fold_failing_unfold (keys : list K) (init : S) :
  merge_fold_failing eqb keys f m1 m2 init
  = let '(out, state) :=
      List.fold_left failing_step keys ([], Some init)
    in
    option_map (fun state => (out, state)) state.
Proof. reflexivity. Qed.

Lemma failing_fold_none (keys : list K) (acc : Keyed.finite_map K C) :
  List.fold_left failing_step keys (acc, None) = (acc, None).
Proof. induction keys as [ | key keys IH ]; [ reflexivity | exact IH ]. Qed.

Lemma not_in_lookup_none {X : Type} (key : K) (l : Keyed.finite_map K X) :
  key ∉ map fst l -> Keyed.lookup eqb key l = None.
Proof.
  intros Hnot. destruct (Keyed.lookup eqb key l) as [ x | ] eqn:Hlookup;
    [ | reflexivity ].
  exfalso. apply Hnot. apply in_map_iff. exists (key, x).
  split; [ reflexivity | exact (keyed_lookup_in eqb Heqb key x l Hlookup) ].
Qed.

Variable Inv : S -> Prop.
Variable le : S -> S -> Prop.
Variable P : K -> option C -> S -> Prop.

Hypothesis Hle_refl : ∀ s : S, le s s.
Hypothesis Hle_trans : ∀ s1 s2 s3 : S, le s1 s2 -> le s2 s3 -> le s1 s3.
Hypothesis Hstep :
  ∀ (key : K) (s s' : S) (c : option C),
    Inv s ->
    f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2) s
    = Some (c, s') ->
    Inv s' ∧ le s s' ∧ P key c s'.

Lemma failing_fold_spec (keys : list K) :
  ∀ (acc out : Keyed.finite_map K C) (s0 staten : S),
    NoDup (map fst acc ++ keys) ->
    Inv s0 ->
    List.fold_left failing_step keys (acc, Some s0) = (out, Some staten) ->
    NoDup (map fst out)
    ∧ Inv staten
    ∧ le s0 staten
    ∧ (∀ key : K,
         key ∈ keys ->
         ∃ st : S, le st staten ∧ P key (Keyed.lookup eqb key out) st)
    ∧ ∀ key : K,
        key ∉ keys -> Keyed.lookup eqb key out = Keyed.lookup eqb key acc.
Proof.
  induction keys as [ | key keys IH ]; intros acc out s0 staten Hnodup Hinv
    Hfold.
  - cbn in Hfold. injection Hfold as <- <-. rewrite app_nil_r in Hnodup.
    split; [ exact Hnodup | ]. split; [ exact Hinv | ].
    split; [ apply Hle_refl | ]. split; [ intros key [] | reflexivity ].
  - cbn [List.fold_left] in Hfold. unfold failing_step at 2 in Hfold.
    cbn [fst snd] in Hfold.
    destruct (f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2) s0)
      as [ [c s1] | ] eqn:Hf.
    2:{ rewrite failing_fold_none in Hfold. discriminate Hfold. }
    destruct (Hstep key s0 s1 c Hinv Hf) as (Hinv1 & Hle1 & Hp).
    assert (Hfresh : key ∉ map fst acc).
    { intros Hin. apply NoDup_remove_2 in Hnodup. apply Hnodup.
      apply in_or_app. left. exact Hin. }
    assert (Hnot_later : key ∉ keys).
    { intros Hin. apply NoDup_remove_2 in Hnodup. apply Hnodup.
      apply in_or_app. right. exact Hin. }
    set (acc1 :=
      match c with
      | Some c => acc ++ [(key, c)]
      | None => acc
      end) in *.
    assert (Hnodup1 : NoDup (map fst acc1 ++ keys)).
    { unfold acc1. destruct c as [ c | ].
      - rewrite map_app, <- app_assoc. exact Hnodup.
      - exact (NoDup_remove_1 _ _ _ Hnodup). }
    assert (Hlookup1 : Keyed.lookup eqb key acc1 = c).
    { unfold acc1. destruct c as [ c | ].
      - rewrite keyed_lookup_app, (not_in_lookup_none key acc Hfresh). cbn.
        rewrite (keyed_eqb_refl eqb Heqb). reflexivity.
      - exact (not_in_lookup_none key acc Hfresh). }
    assert (Hother :
      ∀ key' : K,
        key' <> key ->
        Keyed.lookup eqb key' acc1 = Keyed.lookup eqb key' acc).
    { intros key' Hne. unfold acc1. destruct c as [ c | ]; [ | reflexivity ].
      rewrite keyed_lookup_app.
      destruct (Keyed.lookup eqb key' acc); [ reflexivity | ]. cbn.
      destruct (eqb key' key) eqn:Htest; [ | reflexivity ].
      apply Heqb in Htest. contradiction. }
    destruct (IH acc1 out s1 staten Hnodup1 Hinv1 Hfold)
      as (Hnodup_out & Hinv_n & Hle_n & Hkeys & Hrest).
    split; [ exact Hnodup_out | ]. split; [ exact Hinv_n | ].
    split; [ exact (Hle_trans _ _ _ Hle1 Hle_n) | ]. split.
    + intros key' [ <- | Hin ]; [ | exact (Hkeys key' Hin) ].
      exists s1. split; [ exact Hle_n | ].
      rewrite (Hrest key Hnot_later), Hlookup1. exact Hp.
    + intros key' Hnot. rewrite (Hrest key' (fun H => Hnot (or_intror H))).
      apply Hother. intros ->. apply Hnot. left. reflexivity.
Qed.

(* What [merge_fold_failing] answers, over keys that are different. *)
Lemma merge_fold_failing_spec (keys : list K) (init staten : S)
    (out : Keyed.finite_map K C) :
  NoDup keys ->
  Inv init ->
  merge_fold_failing eqb keys f m1 m2 init = Some (out, staten) ->
  NoDup (map fst out)
  ∧ Inv staten
  ∧ le init staten
  ∧ (∀ key : K,
       key ∈ keys ->
       ∃ st : S, le st staten ∧ P key (Keyed.lookup eqb key out) st)
  ∧ ∀ key : K, key ∉ keys -> Keyed.lookup eqb key out = None.
Proof.
  intros Hnodup Hinv Hmerge. rewrite merge_fold_failing_unfold in Hmerge.
  destruct (List.fold_left failing_step keys ([], Some init))
    as [out' [ state | ]] eqn:Hfold; [ | discriminate Hmerge ].
  cbn in Hmerge. injection Hmerge as <- <-.
  exact (failing_fold_spec keys [] out' init state Hnodup Hinv Hfold).
Qed.

End Merge.

(* ======================================================================== *)
(* [meet_row_like], in parts                                                *)
(* ======================================================================== *)

Section Row.

Context {lattice shape maps_to row_tag known : Type}.

Let case : Type := row_like_case lattice shape maps_to env_extension.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.
Variable meet_maps_to :
  Meet_env.t -> maps_to -> maps_to -> option (Meet_result.t maps_to).
Variable equal_index : lattice -> lattice -> bool.
Variable subset_index : lattice -> lattice -> bool.
Variable union_index : lattice -> lattice -> lattice.
Variable meet_shape : shape -> shape -> Or_bottom.t shape.

Definition with_extra (state : row_like_state)
    (extra : list (variable * flambda_kind)) : row_like_state :=
  {| extra_variables := extra;
     result_env := result_env state;
     result_is_t1 := result_is_t1 state;
     result_is_t2 := result_is_t2 state;
     avoid := avoid state |}.

Definition row_add_extra (common_scope : Scope.t) (scoped_env : Typing_env.t)
    (state : row_like_state) : option (env_extension * row_like_state) :=
  let* level := Typing_env.cut scoped_env common_scope in
  let* extra :=
    union_extra_variables (extra_variables state)
      (rev (Typing_env_level.defined_variables_with_kinds level))
  in
  Some (extension_of_level level, with_extra state extra).

Definition row_join_all (initial_env : Meet_env.t) (common_scope : Scope.t)
    (initial_tenv : Typing_env.t) (scoped_envs : list Typing_env.t)
    (state : row_like_state) : option Meet_env.t :=
  let* result_env :=
    join mh common_scope
      (Meet_env.with_names_to_avoid initial_env (avoid state))
      initial_tenv scoped_envs
  in
  List.fold_left
    (fun env binding =>
       let* env := env in
       define_in_types choices env (fst binding) (snd binding))
    (extra_variables state) (Some result_env).

Definition row_join_result_env (need_join : bool) (scoped_env : Typing_env.t)
    (names : list variable) (state : row_like_state)
    : option row_like_state :=
  let* new_result_env :=
    match result_env state with
    | None => Some [scoped_env]
    | Some other_envs =>
      if need_join then Some (scoped_env :: other_envs) else None
    end
  in
  Some
    {| extra_variables := extra_variables state;
       result_env := Some new_result_env;
       result_is_t1 := result_is_t1 state;
       result_is_t2 := result_is_t2 state;
       avoid := names |}.

Definition row_meet_index (env : Meet_env.t)
    (i1 i2 : row_like_index lattice shape)
    : Meet_result.t (row_like_index lattice shape) :=
  match meet_shape (index_shape i1) (index_shape i2) with
  | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
  | Or_bottom.Ok shape =>
    match index_domain i1, index_domain i2 with
    | Row_like_index_domain.Known i1', Row_like_index_domain.Known i2' =>
      if equal_index i1' i2'
      then Meet_result.Ok Both_inputs env
      else Meet_result.Bottom (New_result tt)
    | Row_like_index_domain.Known known,
      Row_like_index_domain.At_least at_least =>
      if subset_index at_least known
      then Meet_result.Ok Left_input env
      else Meet_result.Bottom (New_result tt)
    | Row_like_index_domain.At_least at_least,
      Row_like_index_domain.Known known =>
      if subset_index at_least known
      then Meet_result.Ok Right_input env
      else Meet_result.Bottom (New_result tt)
    | Row_like_index_domain.At_least i1',
      Row_like_index_domain.At_least i2' =>
      if subset_index i1' i2'
      then
        if subset_index i2' i1'
        then Meet_result.Ok Both_inputs env
        else Meet_result.Ok Right_input env
      else if subset_index i2' i1'
      then Meet_result.Ok Left_input env
      else
        let domain :=
          Row_like_index_domain.At_least (union_index i1' i2')
        in
        Meet_result.Ok (New_result (Row_like_index domain shape)) env
    end
  end.

Definition row_add_case_extension (base_env : Meet_env.t)
    (extension : env_extension) (state : row_like_state)
    : option (Or_bottom.t Meet_env.t) :=
  let* added :=
    Meet_env.add_env_extension_strict
      (Meet_env.with_names_to_avoid base_env (avoid state))
      extension mh
  in
  match added with
  | Or_bottom.Bottom => Some Or_bottom.Bottom
  | Or_bottom.Ok env =>
    match Meet_env.final_typing_env_strict env with
    | Or_bottom.Bottom => Some Or_bottom.Bottom
    | Or_bottom.Ok _ => Some (Or_bottom.Ok env)
    end
  end.

(* What [meet_case] does once the environment of the case is made. *)
Definition row_case_result (common_scope : Scope.t) (need_join : bool)
    (case1 case2 : case)
    (index_result : meet_return_value (row_like_index lattice shape))
    (maps_to_result : meet_return_value maps_to) (env : Meet_env.t)
    (state : row_like_state)
    : option (option (Or_unknown.t case) * row_like_state) :=
  let tenv := Meet_env.final_typing_env env in
  let* state :=
    row_join_result_env need_join tenv (Meet_env.names_to_avoid env) state
  in
  let state := update_refs index_result state in
  let state := update_refs maps_to_result state in
  let index :=
    extract_value index_result (case_index case1) (case_index case2)
  in
  let maps_to :=
    extract_value maps_to_result (case_maps_to case1) (case_maps_to case2)
  in
  let* '(env_extension, state) :=
    if need_join
    then row_add_extra common_scope tenv state
    else if lift_variables_of_single_case choices
    then
      let* '(_, state) := row_add_extra common_scope tenv state in
      Some (empty_env_extension, state)
    else Some (empty_env_extension, state)
  in
  let state :=
    match env_extension with
    | Env_extension [] => state
    | Env_extension (_ :: _) => not_t1 (not_t2 state)
    end
  in
  let case := Row_like_case maps_to index env_extension in
  Some (Some (Or_unknown.Known case), state).

Definition row_meet_case (common_scope : Scope.t) (need_join : bool)
    (env : Meet_env.t) (case1 case2 : case) (state : row_like_state)
    : option (option (Or_unknown.t case) * row_like_state) :=
  let env := Meet_env.with_names_to_avoid env (avoid state) in
  match row_meet_index env (case_index case1) (case_index case2) with
  | Meet_result.Bottom r => Some (None, update_refs r state)
  | Meet_result.Ok index_result env =>
    let* maps_to :=
      meet_maps_to env (case_maps_to case1) (case_maps_to case2)
    in
    match maps_to with
    | Meet_result.Bottom r => Some (None, update_refs r state)
    | Meet_result.Ok maps_to_result env =>
      let* env :=
        Meet_env.add_env_extension_strict env (case_env_extension case1) mh
      in
      let* env :=
        match env with
        | Or_bottom.Bottom => Some Or_bottom.Bottom
        | Or_bottom.Ok env =>
          Meet_env.add_env_extension_strict env (case_env_extension case2) mh
        end
      in
      let env :=
        match env with
        | Or_bottom.Bottom => Or_bottom.Bottom
        | Or_bottom.Ok env =>
          match Meet_env.final_typing_env_strict env with
          | Or_bottom.Bottom => Or_bottom.Bottom
          | Or_bottom.Ok _ => Or_bottom.Ok env
          end
        end
      in
      match env with
      | Or_bottom.Bottom =>
        Some (None, update_refs (@New_result unit tt) state)
      | Or_bottom.Ok env =>
        row_case_result common_scope need_join case1 case2 index_result
          maps_to_result env state
      end
    end
  end.

Definition row_meet_case_with_unknown (need_join : bool)
    (base_env : Meet_env.t) (other_case : case)
    (change : row_like_state -> row_like_state) (state : row_like_state)
    : option (option (Or_unknown.t case) * row_like_state) :=
  let* env :=
    row_add_case_extension base_env (case_env_extension other_case) state
  in
  match env with
  | Or_bottom.Bottom => Some (None, state)
  | Or_bottom.Ok env =>
    let* state :=
      row_join_result_env need_join (Meet_env.final_typing_env env)
        (Meet_env.names_to_avoid env) state
    in
    Some (Some (Or_unknown.Known other_case), change state)
  end.

Definition row_meet_knowns (common_scope : Scope.t) (need_join : bool)
    (base_env : Meet_env.t) (other1 other2 : Or_bottom.t case)
    (case1 case2 : option (Or_unknown.t case)) (state : row_like_state)
    : option (option (Or_unknown.t case) * row_like_state) :=
  match case1, case2 with
  | None, None => Some (None, state)
  | Some case1, None =>
    match other2 with
    | Or_bottom.Bottom => Some (None, not_t1 state)
    | Or_bottom.Ok other_case =>
      match case1 with
      | Or_unknown.Unknown =>
        row_meet_case_with_unknown need_join base_env other_case
          (fun state => not_t1 (not_t2 state))
          state
      | Or_unknown.Known case1 =>
        row_meet_case common_scope need_join base_env case1 other_case state
      end
    end
  | None, Some case2 =>
    match other1 with
    | Or_bottom.Bottom => Some (None, not_t2 state)
    | Or_bottom.Ok other_case =>
      match case2 with
      | Or_unknown.Unknown =>
        row_meet_case_with_unknown need_join base_env other_case
          (fun state => not_t1 (not_t2 state))
          state
      | Or_unknown.Known case2 =>
        row_meet_case common_scope need_join base_env other_case case2 state
      end
    end
  | Some case1, Some case2 =>
    match case1, case2 with
    | Or_unknown.Unknown, Or_unknown.Unknown =>
      let* state :=
        row_join_result_env need_join (Meet_env.final_typing_env base_env)
          (avoid state) state
      in
      Some (Some Or_unknown.Unknown, state)
    | Or_unknown.Known case, Or_unknown.Unknown =>
      row_meet_case_with_unknown need_join base_env case not_t2 state
    | Or_unknown.Unknown, Or_unknown.Known case =>
      row_meet_case_with_unknown need_join base_env case not_t1 state
    | Or_unknown.Known case1, Or_unknown.Known case2 =>
      row_meet_case common_scope need_join base_env case1 case2 state
    end
  end.

Definition row_need_join (is_empty_map_known : known -> bool)
    (get_singleton_map_known : known -> option (row_tag * case))
    (known1 known2 : known) (other1 other2 : Or_bottom.t case) : bool :=
  match
    other1, get_singleton_map_known known1, other2,
    get_singleton_map_known known2
  with
  | Or_bottom.Bottom, Some _, _, _ | _, _, Or_bottom.Bottom, Some _ => false
  | (Or_bottom.Ok _ | Or_bottom.Bottom), _,
    (Or_bottom.Ok _ | Or_bottom.Bottom), _ =>
    if is_empty_map_known known1 && is_empty_map_known known2
    then false
    else true
  end.

Definition row_initial_state (base_env : Meet_env.t) : row_like_state :=
  {| extra_variables := [];
     result_env := None;
     result_is_t1 := true;
     result_is_t2 := true;
     avoid := Meet_env.names_to_avoid base_env |}.

Definition row_match_with_input {A : Type} (state : row_like_state) (v : A)
    : meet_return_value A :=
  match result_is_t1 state, result_is_t2 state with
  | true, true => Both_inputs
  | true, false => Left_input
  | false, true => Right_input
  | false, false => New_result v
  end.

(* The pair of the cases of the other tags. *)
Definition row_meet_others (common_scope : Scope.t) (need_join : bool)
    (base_env : Meet_env.t) (other1 other2 : Or_bottom.t case)
    (state : row_like_state) : option (Or_bottom.t case * row_like_state) :=
  match other1, other2 with
  | Or_bottom.Bottom, Or_bottom.Bottom => Some (Or_bottom.Bottom, state)
  | Or_bottom.Bottom, _ => Some (Or_bottom.Bottom, not_t2 state)
  | _, Or_bottom.Bottom => Some (Or_bottom.Bottom, not_t1 state)
  | Or_bottom.Ok other1, Or_bottom.Ok other2 =>
    let* '(other, state) :=
      row_meet_case common_scope need_join base_env other1 other2 state
    in
    match other with
    | None => Some (Or_bottom.Bottom, state)
    | Some Or_unknown.Unknown => None
    | Some (Or_unknown.Known r) => Some (Or_bottom.Ok r, state)
    end
  end.

(* What [meet_row_like] answers, the cases met. *)
Definition row_finish (is_empty_map_known : known -> bool)
    (initial_env : Meet_env.t) (common_scope : Scope.t)
    (initial_tenv : Typing_env.t) (cases : known) (other : Or_bottom.t case)
    (state : row_like_state)
    : option (Meet_result.t (known * Or_bottom.t case)) :=
  if is_empty_map_known cases
     && match other with
        | Or_bottom.Bottom => true
        | Or_bottom.Ok _ => false
        end
  then Some (Meet_result.Bottom (New_result tt))
  else
    let* env :=
      match result_env state with
      | None => Some Or_bottom.Bottom
      | Some scoped_envs =>
        let* env :=
          row_join_all initial_env common_scope initial_tenv scoped_envs state
        in
        Some (Or_bottom.Ok env)
      end
    in
    match env with
    | Or_bottom.Bottom =>
      Some (Meet_result.Bottom (row_match_with_input state tt))
    | Or_bottom.Ok env =>
      Some (Meet_result.Ok (row_match_with_input state (cases, other)) env)
    end.

Lemma meet_row_like_unfold (is_empty_map_known : known -> bool)
    (get_singleton_map_known : known -> option (row_tag * case))
    (merge_map_known :
      (row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
       row_like_state -> option (option (Or_unknown.t case) * row_like_state))
      -> known -> known -> row_like_state -> option (known * row_like_state))
    (initial_env : Meet_env.t) (known1 known2 : known)
    (other1 other2 : Or_bottom.t case) :
  meet_row_like join choices mh meet_maps_to equal_index subset_index
    union_index meet_shape is_empty_map_known get_singleton_map_known
    merge_map_known initial_env known1 known2 other1 other2
  = let common_scope :=
      Typing_env.current_scope (Meet_env.typing_env initial_env)
    in
    let initial_tenv := Meet_env.typing_env initial_env in
    let base_env := snd (Meet_env.enter_scope initial_env) in
    let need_join :=
      row_need_join is_empty_map_known get_singleton_map_known known1 known2
        other1 other2
    in
    let* '(cases, state) :=
      merge_map_known
        (fun _tag case1 case2 =>
           row_meet_knowns common_scope need_join base_env other1 other2 case1
             case2)
        known1 known2 (row_initial_state base_env)
    in
    let* '(other, state) :=
      row_meet_others common_scope need_join base_env other1 other2 state
    in
    row_finish is_empty_map_known initial_env common_scope initial_tenv cases
      other state.
Proof. reflexivity. Qed.


(* ======================================================================== *)
(* What is supposed of the parts                                            *)
(* ======================================================================== *)

Hypothesis Hjoin : join_scoped_envs_sound join.
Hypothesis Hdefine : define_again choices = false.
Hypothesis Hlift : lift_variables_of_single_case choices = true.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

Let index : Type := row_like_index lattice shape.

(* The index and the product of a case are well formed together, for
   the case of a tag or for the case of the other tags; a product fits
   an environment; the index, and the product, are true of the value in
   hand; and the product is true and the value has every position of
   it. *)
Variable index_ok : option row_tag -> index -> maps_to -> Prop.
Variable maps_fits : Typing_env.t -> maps_to -> Prop.
Variable index_true : index -> Prop.
Variable maps_true : static_env -> env -> maps_to -> Prop.
Variable maps_holds : static_env -> env -> maps_to -> Prop.

Hypothesis Hmaps_stable : part_stable maps_fits maps_holds.
Hypothesis Hmaps_kinds : fits_by_kinds maps_fits.
Hypothesis Hmaps_true_stable :
  ∀ (statics : static_env) (tenv : Typing_env.t) (valuation valuation' : env)
    (m : maps_to),
    maps_fits tenv m ->
    extended_from tenv valuation valuation' ->
    maps_true statics valuation m ->
    maps_true statics valuation' m.
Hypothesis Hmeet_maps : part_meet_ok maps_fits maps_holds meet_maps_to.
Hypothesis Hholds :
  ∀ (statics : static_env) (valuation : env) (o : option row_tag) (i : index)
    (m : maps_to),
    statics_valid statics ->
    index_ok o i m ->
    index_true i ->
    maps_true statics valuation m ->
    maps_holds statics valuation m.
Hypothesis Htrue :
  ∀ (statics : static_env) (valuation : env) (m : maps_to),
    maps_holds statics valuation m -> maps_true statics valuation m.
Hypothesis Hindex_other :
  ∀ (tag : row_tag) (i : index) (m : maps_to),
    index_ok None i m -> index_ok (Some tag) i m.
Hypothesis Hindex_ok :
  ∀ (o : option row_tag) (menv0 menv1 menv2 : Meet_env.t) (i1 i2 : index)
    (m1 m2 : maps_to) (ri : meet_return_value index)
    (rm : meet_return_value maps_to),
    index_ok o i1 m1 ->
    index_ok o i2 m2 ->
    row_meet_index menv0 i1 i2 = Meet_result.Ok ri menv1 ->
    meet_maps_to menv1 m1 m2 = Some (Meet_result.Ok rm menv2) ->
    index_ok o (extract_value ri i1 i2) (extract_value rm m1 m2).
Hypothesis Hindex_true :
  ∀ (o : option row_tag) (menv0 : Meet_env.t) (i1 i2 : index)
    (m1 m2 : maps_to),
    index_ok o i1 m1 ->
    index_ok o i2 m2 ->
    index_true i1 ->
    index_true i2 ->
    ∃ ri : meet_return_value index,
      row_meet_index menv0 i1 i2 = Meet_result.Ok ri menv0
      ∧ index_true (extract_value ri i1 i2).

(* A case fits an environment, and is true of the value in hand. *)
Definition case_fits (tenv : Typing_env.t) (o : option row_tag) (c : case)
    : Prop :=
  index_ok o (case_index c) (case_maps_to c)
  ∧ maps_fits tenv (case_maps_to c)
  ∧ extension_fits tenv (case_env_extension c).

Definition case_true (statics : static_env) (valuation : env) (c : case)
    : Prop :=
  index_true (case_index c)
  ∧ maps_true statics valuation (case_maps_to c)
  ∧ env_extension_satisfied statics valuation (case_env_extension c).

Lemma case_fits_kept (tenv tenv' : Typing_env.t) (o : option row_tag)
    (c : case) :
  knows_at_kinds tenv tenv' -> case_fits tenv o c -> case_fits tenv' o c.
Proof.
  intros Hknows (Hindex & Hmaps & Hextension).
  split; [ exact Hindex | ].
  split; [ exact (Hmaps_kinds tenv tenv' _ Hknows Hmaps) | ].
  exact (extension_fits_kept tenv tenv' _ Hknows Hextension).
Qed.

Lemma case_true_stable (statics : static_env) (tenv : Typing_env.t)
    (valuation valuation' : env) (o : option row_tag) (c : case) :
  case_fits tenv o c ->
  extended_from tenv valuation valuation' ->
  case_true statics valuation c ->
  case_true statics valuation' c.
Proof.
  intros (_ & Hmaps & _ & Hextension & _) Hextended (Hi & Hm & He).
  split; [ exact Hi | ].
  split; [ exact (Hmaps_true_stable statics tenv valuation valuation' _ Hmaps
                    Hextended Hm) | ].
  exact (env_extension_satisfied_extended statics tenv valuation valuation' _
           Hextension Hextended He).
Qed.

(* What a tag selects of a row: its case, or the case of the other
   tags. It is true of the value in hand if it is unknown, or a case
   that is true. *)
Definition selected (c : option (Or_unknown.t case)) (other : Or_bottom.t case)
    : option (Or_unknown.t case) :=
  match c with
  | Some c => Some c
  | None =>
    match other with
    | Or_bottom.Ok c => Some (Or_unknown.Known c)
    | Or_bottom.Bottom => None
    end
  end.

Definition sel_true (statics : static_env) (valuation : env)
    (s : option (Or_unknown.t case)) : Prop :=
  match s with
  | None => False
  | Some Or_unknown.Unknown => True
  | Some (Or_unknown.Known c) => case_true statics valuation c
  end.

Definition sel_fits (tenv : Typing_env.t) (o : option row_tag)
    (s : option (Or_unknown.t case)) : Prop :=
  match s with
  | Some (Or_unknown.Known c) => case_fits tenv o c
  | _ => True
  end.

Lemma sel_true_stable (statics : static_env) (tenv : Typing_env.t)
    (valuation valuation' : env) (o : option row_tag)
    (s : option (Or_unknown.t case)) :
  sel_fits tenv o s ->
  extended_from tenv valuation valuation' ->
  sel_true statics valuation s ->
  sel_true statics valuation' s.
Proof.
  destruct s as [ [ c | ] | ]; cbn; [ | auto | auto ].
  apply case_true_stable.
Qed.

(* ======================================================================== *)
(* The state of the fold                                                    *)
(* ======================================================================== *)

Variable menv : Meet_env.t.

Hypothesis Hfits : typing_env_fits (Meet_env.typing_env menv).

Let tenv : Typing_env.t := Meet_env.typing_env menv.
Let base : Meet_env.t := snd (Meet_env.enter_scope menv).
Let common_scope : Scope.t := Typing_env.current_scope tenv.

Variable need_join : bool.

Definition envs_of (state : row_like_state) : list Typing_env.t :=
  match result_env state with
  | None => []
  | Some envs => envs
  end.

Definition level_of (stenv : Typing_env.t) : Typing_env_level.t :=
  Typing_env.One_level.level (Typing_env.current_level stenv).

(* The environments kept share only what the environment of the meet
   knows. *)
Inductive envs_apart : list Typing_env.t -> Prop :=
  | Apart_nil : envs_apart []
  | Apart_cons (e : Typing_env.t) (l : list Typing_env.t) :
    envs_apart l ->
    (∀ (e' : Typing_env.t) (x : variable),
       e' ∈ l ->
       Typing_env.mem None e (Name_var x) = true ->
       Typing_env.mem None e' (Name_var x) = true ->
       Typing_env.mem None tenv (Name_var x) = true) ->
    envs_apart (e :: l).

Lemma envs_apart_in (l : list Typing_env.t) :
  envs_apart l ->
  ∀ e e' : Typing_env.t,
    e ∈ l ->
    e' ∈ l ->
    e = e'
    ∨ ∀ x : variable,
        Typing_env.mem None e (Name_var x) = true ->
        Typing_env.mem None e' (Name_var x) = true ->
        Typing_env.mem None tenv (Name_var x) = true.
Proof.
  induction 1 as [ | a l Hl IH Ha ]; intros e e' Hin Hin'; [ destruct Hin | ].
  destruct Hin as [ <- | Hin ], Hin' as [ <- | Hin' ].
  - left. reflexivity.
  - right. intros x Hx Hx'. exact (Ha e' x Hin' Hx Hx').
  - right. intros x Hx Hx'. exact (Ha e x Hin Hx' Hx).
  - exact (IH e e' Hin Hin').
Qed.

Record row_inv (state : row_like_state) : Prop := {
  inv_scoped :
    ∀ stenv : Typing_env.t,
      stenv ∈ envs_of state ->
      tenv_extends (Typing_env.increment_scope tenv) stenv
      ∧ typing_env_fits stenv;
  inv_apart : envs_apart (envs_of state);
  inv_new :
    ∀ (stenv : Typing_env.t) (x : variable),
      stenv ∈ envs_of state ->
      Typing_env.mem None stenv (Name_var x) = true ->
      Typing_env.mem None tenv (Name_var x) = false ->
      among x (Meet_env.names_to_avoid menv) = false
      ∧ among x (avoid state) = true;
  inv_avoid :
    ∀ x : variable,
      among x (Meet_env.names_to_avoid menv) = true ->
      among x (avoid state) = true;
  inv_one_kind : one_kind (extra_variables state);
  inv_extra :
    ∀ (y : variable) (k : flambda_kind),
      lookup y (extra_variables state) = Some k ->
      ∃ stenv : Typing_env.t,
        stenv ∈ envs_of state
        ∧ Typing_env_level.find_kind (level_of stenv) y = Some k
}.

(* A later state has the environments and the extra variables of an
   earlier one. *)
Definition state_le (s s' : row_like_state) : Prop :=
  (∀ e : Typing_env.t, e ∈ envs_of s -> e ∈ envs_of s')
  ∧ ∀ (y : variable) (k : flambda_kind),
      lookup y (extra_variables s) = Some k ->
      lookup y (extra_variables s') = Some k.

Lemma state_le_refl (s : row_like_state) : state_le s s.
Proof. split; auto. Qed.

Lemma state_le_trans (s1 s2 s3 : row_like_state) :
  state_le s1 s2 -> state_le s2 s3 -> state_le s1 s3.
Proof. intros [H1 H1'] [H2 H2']. split; auto. Qed.

(* The flags of the state are no part of this. *)
Definition same_but_flags (s s' : row_like_state) : Prop :=
  extra_variables s' = extra_variables s
  ∧ result_env s' = result_env s
  ∧ avoid s' = avoid s.

Lemma same_but_flags_inv (s s' : row_like_state) :
  same_but_flags s s' -> row_inv s -> row_inv s'.
Proof.
  intros (Hextra & Henvs & Havoid) [H1 H2 H3 H4 H5 H6].
  unfold envs_of in *.
  split; unfold envs_of; rewrite ?Hextra, ?Henvs, ?Havoid; assumption.
Qed.

Lemma same_but_flags_le (s s' : row_like_state) :
  same_but_flags s s' -> state_le s s' ∧ state_le s' s.
Proof.
  intros (Hextra & Henvs & Havoid). unfold state_le, envs_of.
  rewrite Hextra, Henvs. auto.
Qed.

Lemma update_refs_same {A : Type} (r : meet_return_value A)
    (s : row_like_state) :
  same_but_flags s (update_refs r s).
Proof. destruct r; repeat split. Qed.

Lemma not_t1_same (s : row_like_state) : same_but_flags s (not_t1 s).
Proof. repeat split. Qed.

Lemma not_t2_same (s : row_like_state) : same_but_flags s (not_t2 s).
Proof. repeat split. Qed.

Lemma same_but_flags_trans (s1 s2 s3 : row_like_state) :
  same_but_flags s1 s2 -> same_but_flags s2 s3 -> same_but_flags s1 s3.
Proof.
  intros (H1 & H2 & H3) (H1' & H2' & H3').
  repeat split; congruence.
Qed.

Lemma row_meet_index_env (env env' : Meet_env.t) (i1 i2 : index)
    (r : meet_return_value index) :
  row_meet_index env i1 i2 = Meet_result.Ok r env' -> env' = env.
Proof.
  unfold row_meet_index.
  destruct (meet_shape (index_shape i1) (index_shape i2)); [ | discriminate ].
  destruct (index_domain i1), (index_domain i2);
    repeat match goal with
           | |- context [if ?b then _ else _] => destruct b
           end;
    try discriminate; intros [= _ <-]; reflexivity.
Qed.

(* ======================================================================== *)
(* An environment is kept                                                   *)
(* ======================================================================== *)

Lemma row_join_result_env_spec (stenv : Typing_env.t) (names : list variable)
    (state state1 : row_like_state) :
  row_join_result_env need_join stenv names state = Some state1 ->
  envs_of state1 = stenv :: envs_of state
  ∧ extra_variables state1 = extra_variables state
  ∧ avoid state1 = names.
Proof.
  unfold row_join_result_env, envs_of.
  destruct (result_env state) as [ others | ];
    [ destruct need_join; [ | discriminate ] | ];
    cbn; intros [= <-]; cbn; auto.
Qed.

(* The environment of a case that stays is kept. It was made from the
   base with the names to avoid of the state. *)
Lemma row_push (scoped : Meet_env.t) (state state1 : row_like_state) :
  row_inv state ->
  meet_env_extends (Meet_env.with_names_to_avoid base (avoid state)) scoped ->
  typing_env_fits (Meet_env.typing_env scoped) ->
  new_names_avoided (Meet_env.with_names_to_avoid base (avoid state)) scoped ->
  row_join_result_env need_join (Meet_env.typing_env scoped)
    (Meet_env.names_to_avoid scoped) state
  = Some state1 ->
  row_inv state1
  ∧ state_le state state1
  ∧ Meet_env.typing_env scoped ∈ envs_of state1
  ∧ extra_variables state1 = extra_variables state.
Proof.
  intros [Hscoped Hapart Hnew Havoid Hone Hextra] Hextends Hscoped_fits Hnames
    Hpush.
  destruct (row_join_result_env_spec _ _ _ _ Hpush)
    as (Henvs & Hextra1 & Havoid1).
  assert (Hmore :
    ∀ x : variable,
      among x (avoid state) = true ->
      among x (Meet_env.names_to_avoid scoped) = true).
  { exact (meet_env_extends_avoid _ _ Hextends). }
  split; [ | split; [ | split ] ].
  - split.
    + intros stenv Hin. rewrite Henvs in Hin.
      destruct Hin as [ <- | Hin ]; [ | exact (Hscoped stenv Hin) ].
      split; [ | exact Hscoped_fits ].
      exact (meet_env_extends_tenv _ _ Hextends).
    + rewrite Henvs. apply Apart_cons; [ exact Hapart | ].
      intros e' x Hin Hx Hx'.
      destruct (Typing_env.mem None tenv (Name_var x)) eqn:Hold;
        [ reflexivity | exfalso ].
      destruct (Hnew e' x Hin Hx' Hold) as [_ Hamong].
      destruct (Hnames x Hx Hold) as [Hnot _].
      change (among x (avoid state) = false) in Hnot.
      rewrite Hamong in Hnot. discriminate Hnot.
    + intros stenv x Hin Hx Hold. rewrite Henvs in Hin. rewrite Havoid1.
      destruct Hin as [ <- | Hin ].
      * destruct (Hnames x Hx Hold) as (Hnot & Hyes & _).
        split; [ | exact Hyes ].
        change (among x (avoid state) = false) in Hnot.
        destruct (among x (Meet_env.names_to_avoid menv)) eqn:Hmenv;
          [ | reflexivity ].
        rewrite (Havoid x Hmenv) in Hnot. discriminate Hnot.
      * destruct (Hnew stenv x Hin Hx Hold) as [Hfresh Hamong].
        split; [ exact Hfresh | exact (Hmore x Hamong) ].
    + intros x Hx. rewrite Havoid1. exact (Hmore x (Havoid x Hx)).
    + rewrite Hextra1. exact Hone.
    + intros y k Hy. rewrite Hextra1 in Hy.
      destruct (Hextra y k Hy) as (stenv & Hin & Hkind).
      exists stenv. split; [ rewrite Henvs; right; exact Hin | exact Hkind ].
  - split.
    + intros e Hin. rewrite Henvs. right. exact Hin.
    + intros y k Hy. rewrite Hextra1. exact Hy.
  - rewrite Henvs. left. reflexivity.
  - exact Hextra1.
Qed.

(* The variables of the level of an environment that is kept are added
   to the extra variables. *)
Definition covered (stenv : Typing_env.t)
    (extra : list (variable * flambda_kind)) : Prop :=
  ∀ (y : variable) (k : flambda_kind),
    Typing_env_level.find_kind (level_of stenv) y = Some k ->
    lookup y extra = Some k.

Lemma row_add_extra_spec (stenv : Typing_env.t) (state state' : row_like_state)
    (extension : env_extension) :
  row_inv state ->
  stenv ∈ envs_of state ->
  row_add_extra common_scope stenv state = Some (extension, state') ->
  extension = extension_of_level (level_of stenv)
  ∧ row_inv state'
  ∧ state_le state state'
  ∧ envs_of state' = envs_of state
  ∧ covered stenv (extra_variables state').
Proof.
  intros [Hscoped Hapart Hnew Havoid Hone Hextra] Hin Hadd.
  destruct (Hscoped stenv Hin) as [Hextends Hscoped_fits].
  unfold row_add_extra, common_scope in Hadd.
  rewrite (scoped_cut tenv stenv Hextends) in Hadd.
  cbv beta iota in Hadd.
  fold (level_of stenv) in Hadd.
  destruct (union_extra_variables (extra_variables state)
              (rev
                 (Typing_env_level.defined_variables_with_kinds
                    (level_of stenv))))
    as [ extra | ] eqn:Hunion; [ | discriminate Hadd ].
  cbn in Hadd. injection Hadd as <- <-.
  destruct (union_extra_variables_spec _ _ _ Hone Hunion)
    as (Hone' & Hkept & Hvars & Hfrom).
  assert (Hlevel :
    ∀ (y : variable) (k : flambda_kind),
      Typing_env_level.find_kind (level_of stenv) y = Some k ->
      (y, k)
      ∈ rev (Typing_env_level.defined_variables_with_kinds (level_of stenv))).
  { intros y k Hk. apply in_rev. rewrite rev_involutive.
    exact (lookup_in y k _ Hk). }
  split; [ reflexivity | ]. split; [ | split; [ | split ] ].
  - split; try assumption.
    intros y k Hy. change (lookup y extra = Some k) in Hy.
    destruct (Hfrom y k Hy) as [ Hold | Hlevel_y ];
      [ exact (Hextra y k Hold) | ].
    exists stenv. split; [ exact Hin | ].
    apply in_rev in Hlevel_y.
    destruct (in_lookup_some y k _ Hlevel_y) as [k1 Hk1].
    assert (Hsame := Hvars y k1 (Hlevel y k1 Hk1)).
    rewrite Hy in Hsame. injection Hsame as <-. exact Hk1.
  - split; [ auto | exact Hkept ].
  - reflexivity.
  - intros y k Hk. exact (Hvars y k (Hlevel y k Hk)).
Qed.

(* ======================================================================== *)
(* What is kept of a case that was met                                      *)
(* ======================================================================== *)

(* A case that is answered fits the environment of the meet, or one of
   the environments kept, the variables of whose level are among the
   extra variables. *)
Definition case_placed (o : option row_tag) (c : case)
    (state : row_like_state) : Prop :=
  case_fits tenv o c
  ∨ ∃ stenv : Typing_env.t,
      stenv ∈ envs_of state
      ∧ covered stenv (extra_variables state)
      ∧ case_fits stenv o c.

Definition result_placed (o : option row_tag)
    (r : option (Or_unknown.t case)) (state : row_like_state) : Prop :=
  match r with
  | Some (Or_unknown.Known c) => case_placed o c state
  | _ => True
  end.

Lemma result_placed_le (o : option row_tag) (r : option (Or_unknown.t case))
    (s s' : row_like_state) :
  state_le s s' -> result_placed o r s -> result_placed o r s'.
Proof.
  intros [Henvs Hextra]. destruct r as [ [ c | ] | ]; cbn; auto.
  intros [ Hc | (stenv & Hin & Hcovered & Hc) ]; [ left; exact Hc | right ].
  exists stenv. split; [ exact (Henvs stenv Hin) | ].
  split; [ | exact Hc ].
  intros y k Hk. exact (Hextra y k (Hcovered y k Hk)).
Qed.

(* No value is lost where a pair of cases is met: if the two are true
   of the value in hand, there is an environment kept and a valuation
   that satisfies it, and under every valuation that agrees with that
   one on what the environment knows the case answered is true. *)
Definition pair_sound (s1 s2 r : option (Or_unknown.t case))
    (state : row_like_state) : Prop :=
  ∀ (statics : static_env) (valuation : env),
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation tenv ->
    sel_true statics valuation s1 ->
    sel_true statics valuation s2 ->
    ∃ (stenv : Typing_env.t) (valuation_k : env),
      stenv ∈ envs_of state
      ∧ extended_from tenv valuation valuation_k
      ∧ typing_env_satisfied_strictly statics valuation_k stenv
      ∧ keeps_equations_valued (Typing_env.increment_scope tenv) stenv
          valuation valuation_k
      ∧ ∀ valuation' : env,
          extended_from stenv valuation_k valuation' ->
          sel_true statics valuation' r.

Lemma pair_sound_le (s1 s2 r : option (Or_unknown.t case))
    (s s' : row_like_state) :
  state_le s s' -> pair_sound s1 s2 r s -> pair_sound s1 s2 r s'.
Proof.
  intros [Henvs _] Hpair statics valuation Hstatics Hsatisfied H1 H2.
  destruct (Hpair statics valuation Hstatics Hsatisfied H1 H2)
    as (stenv & valuation_k & Hin & Hrest).
  exists stenv, valuation_k. split; [ exact (Henvs stenv Hin) | exact Hrest ].
Qed.

(* What a step of the fold keeps. *)
Definition row_step (o : option row_tag) (s1 s2 : option (Or_unknown.t case))
    (state : row_like_state)
    (answer : option (option (Or_unknown.t case) * row_like_state)) : Prop :=
  ∀ (r : option (Or_unknown.t case)) (state' : row_like_state),
    answer = Some (r, state') ->
    row_inv state'
    ∧ state_le state state'
    ∧ result_placed o r state'
    ∧ pair_sound s1 s2 r state'.

(* The base, with the names to avoid of a state. *)
Lemma base_fits (names : list variable) :
  typing_env_fits
    (Meet_env.typing_env (Meet_env.with_names_to_avoid base names)).
Proof. exact (increment_scope_fits tenv Hfits). Qed.

Lemma base_knows (names : list variable) :
  knows_at_kinds tenv
    (Meet_env.typing_env (Meet_env.with_names_to_avoid base names)).
Proof. exact (increment_scope_knows tenv). Qed.

Lemma base_satisfied (names : list variable) (statics : static_env)
    (valuation : env) :
  typing_env_satisfied_strictly statics valuation tenv ->
  typing_env_satisfied_strictly statics valuation
    (Meet_env.typing_env (Meet_env.with_names_to_avoid base names)).
Proof.
  intros Hsatisfied. apply (increment_scope_satisfied (fun _ => False)).
  exact Hsatisfied.
Qed.

(* Two cases that are unknown: the base itself is kept. *)
Lemma row_unknowns_step (o : option row_tag) (state : row_like_state) :
  row_inv state ->
  row_step o (Some Or_unknown.Unknown) (Some Or_unknown.Unknown) state
    (let* state :=
       row_join_result_env need_join (Meet_env.final_typing_env base)
         (avoid state) state
     in
     Some (Some Or_unknown.Unknown, state)).
Proof.
  intros Hinv r state' Hanswer.
  destruct (row_join_result_env need_join (Meet_env.final_typing_env base)
              (avoid state) state)
    as [ state1 | ] eqn:Hpush; [ | discriminate Hanswer ].
  cbn in Hanswer. injection Hanswer as <- <-.
  destruct (row_push (Meet_env.with_names_to_avoid base (avoid state)) state
              state1 Hinv (meet_env_extends_refl _) (base_fits _)
              (new_names_avoided_refl _) Hpush)
    as (Hinv1 & Hle & Hin & _).
  split; [ exact Hinv1 | ]. split; [ exact Hle | ]. split; [ exact I | ].
  intros statics valuation Hstatics Hsatisfied _ _.
  exists (Typing_env.increment_scope tenv), valuation.
  split; [ exact Hin | ]. split; [ apply extended_from_refl | ].
  split; [ exact (base_satisfied [] statics valuation Hsatisfied) | ].
  split; [ apply keeps_equations_valued_refl | ].
  intros valuation' _. exact I.
Qed.

Lemma same_but_flags_envs (s s' : row_like_state) :
  same_but_flags s s' -> envs_of s' = envs_of s.
Proof. intros (_ & Henvs & _). unfold envs_of. rewrite Henvs. reflexivity. Qed.

Lemma row_step_flags (o : option row_tag) (s1 s2 r : option (Or_unknown.t case))
    (state state1 state' : row_like_state) :
  same_but_flags state1 state' ->
  row_inv state1
  ∧ state_le state state1
  ∧ result_placed o r state1
  ∧ pair_sound s1 s2 r state1 ->
  row_inv state'
  ∧ state_le state state'
  ∧ result_placed o r state'
  ∧ pair_sound s1 s2 r state'.
Proof.
  intros Hsame (Hinv & Hle & Hplaced & Hpair).
  destruct (same_but_flags_le _ _ Hsame) as [Hle' _].
  split; [ exact (same_but_flags_inv _ _ Hsame Hinv) | ].
  split; [ exact (state_le_trans _ _ _ Hle Hle') | ].
  split; [ exact (result_placed_le o r _ _ Hle' Hplaced) | ].
  exact (pair_sound_le s1 s2 r _ _ Hle' Hpair).
Qed.

(* A valuation that agrees with another on what an environment kept
   knows agrees with it on what the environment of the meet knows. *)
Lemma extended_from_scoped (scoped : Meet_env.t) (names : list variable)
    (valuation valuation1 valuation' : env) :
  meet_env_extends (Meet_env.with_names_to_avoid base names) scoped ->
  extended_from tenv valuation valuation1 ->
  extended_from (Meet_env.typing_env scoped) valuation1 valuation' ->
  extended_from tenv valuation valuation'.
Proof.
  intros Hextends Hextended1 Hextended' x Hx.
  rewrite (Hextended' x
             (proj1 (meet_env_extends_knows _ _ Hextends) _
                (proj1 (base_knows names) _ Hx))).
  exact (Hextended1 x Hx).
Qed.

(* A case of one input that is unknown, with a case of the other. *)
Lemma row_with_unknown_step (o : option row_tag) (other_case : case)
    (change : row_like_state -> row_like_state)
    (s1 s2 : option (Or_unknown.t case)) (state : row_like_state) :
  row_inv state ->
  case_fits tenv o other_case ->
  (∀ s : row_like_state, same_but_flags s (change s)) ->
  (∀ (statics : static_env) (valuation : env),
     sel_true statics valuation s1 ->
     sel_true statics valuation s2 ->
     case_true statics valuation other_case) ->
  row_step o s1 s2 state
    (row_meet_case_with_unknown need_join base other_case change state).
Proof.
  intros Hinv Hcase Hchange Hsel r state' Hanswer.
  unfold row_meet_case_with_unknown, row_add_case_extension in Hanswer.
  set (env0 := Meet_env.with_names_to_avoid base (avoid state)) in *.
  assert (Hextension :
    optional_extension_fits (Meet_env.typing_env env0)
      (Some (case_env_extension other_case))).
  { exact (extension_fits_kept tenv _ _ (base_knows _) (proj2 (proj2 Hcase))). }
  destruct (Meet_env.add_env_extension_strict env0
              (case_env_extension other_case) mh)
    as [ [ env1 | ] | ] eqn:Hadd; [ | | discriminate Hanswer ].
  - cbv beta iota delta [Meet_env.final_typing_env_strict] in Hanswer.
    destruct (row_join_result_env need_join (Meet_env.final_typing_env env1)
                (Meet_env.names_to_avoid env1) state)
      as [ state1 | ] eqn:Hpush; [ | discriminate Hanswer ].
    cbv beta iota in Hanswer. injection Hanswer as <- <-.
    destruct (add_optional_extension_keeps mh Hkeeps env0 env1 _ (base_fits _)
                Hextension Hadd)
      as [[Hextends Hfits1] Hnames].
    destruct (row_push env1 state state1 Hinv Hextends Hfits1 Hnames Hpush)
      as (Hinv1 & Hle & Hin & _).
    apply (row_step_flags o s1 s2 _ state state1 _ (Hchange state1)).
    split; [ exact Hinv1 | ]. split; [ exact Hle | ].
    split; [ left; exact Hcase | ].
    intros statics valuation Hstatics Hsatisfied H1 H2.
    assert (Htrue_case := Hsel statics valuation H1 H2).
    destruct (add_optional_extension_sound mh Hkeeps Hsound statics valuation
                env0 _ _ (base_fits _) Hextension Hstatics
                (base_satisfied _ statics valuation Hsatisfied)
                (proj2 (proj2 Htrue_case)) Hadd)
      as (env1' & [= <-] & valuation1 & Hextended1 & Hsatisfied1 & Hvalued1).
    exists (Meet_env.typing_env env1), valuation1.
    split; [ exact Hin | ]. split; [ exact Hextended1 | ].
    split; [ exact Hsatisfied1 | ]. split; [ exact Hvalued1 | ].
    intros valuation' Hextended'.
    exact (case_true_stable statics tenv valuation valuation' o other_case
             Hcase
             (extended_from_scoped env1 _ valuation valuation1 valuation'
                Hextends Hextended1 Hextended')
             Htrue_case).
  - cbv beta iota in Hanswer. injection Hanswer as <- <-.
    split; [ exact Hinv | ]. split; [ apply state_le_refl | ].
    split; [ exact I | ].
    intros statics valuation Hstatics Hsatisfied H1 H2.
    assert (Htrue_case := Hsel statics valuation H1 H2).
    destruct (add_optional_extension_sound mh Hkeeps Hsound statics valuation
                env0 _ _ (base_fits _) Hextension Hstatics
                (base_satisfied _ statics valuation Hsatisfied)
                (proj2 (proj2 Htrue_case)) Hadd)
      as (env1' & Hbottom & _).
    discriminate Hbottom.
Qed.

(* The extension that has no equation. *)
Lemma empty_extension_fits (stenv : Typing_env.t) :
  extension_fits stenv empty_env_extension.
Proof.
  split; [ reflexivity | ]. split; intros n ty Hlookup; discriminate Hlookup.
Qed.

Lemma empty_extension_satisfied (statics : static_env) (valuation : env) :
  env_extension_satisfied statics valuation empty_env_extension.
Proof.
  apply (env_extension_lookup statics valuation). intros n ty Hlookup.
  discriminate Hlookup.
Qed.

(* A case is answered: its environment is kept, and the variables of
   its level are added to the extra variables. *)
Lemma row_case_result_step (o : option row_tag) (c1 c2 : case)
    (ri : meet_return_value index) (rm : meet_return_value maps_to)
    (env4 : Meet_env.t) (state : row_like_state) :
  let index' := extract_value ri (case_index c1) (case_index c2) in
  let maps' := extract_value rm (case_maps_to c1) (case_maps_to c2) in
  row_inv state ->
  meet_env_extends (Meet_env.with_names_to_avoid base (avoid state)) env4 ->
  typing_env_fits (Meet_env.typing_env env4) ->
  new_names_avoided (Meet_env.with_names_to_avoid base (avoid state)) env4 ->
  index_ok o index' maps' ->
  maps_fits (Meet_env.typing_env env4) maps' ->
  (∀ (statics : static_env) (valuation : env),
     statics_valid statics ->
     typing_env_satisfied_strictly statics valuation tenv ->
     case_true statics valuation c1 ->
     case_true statics valuation c2 ->
     ∃ valuation4 : env,
       extended_from tenv valuation valuation4
       ∧ typing_env_satisfied_strictly statics valuation4
           (Meet_env.typing_env env4)
       ∧ keeps_equations_valued (Typing_env.increment_scope tenv)
           (Meet_env.typing_env env4) valuation valuation4
       ∧ index_true index'
       ∧ maps_holds statics valuation4 maps') ->
  row_step o (Some (Or_unknown.Known c1)) (Some (Or_unknown.Known c2)) state
    (row_case_result common_scope need_join c1 c2 ri rm env4 state).
Proof.
  intros index' maps' Hinv Hextends Hfits4 Hnames Hindex Hmaps Hlost r state'
    Hanswer.
  unfold row_case_result in Hanswer.
  change (Meet_env.final_typing_env env4) with (Meet_env.typing_env env4)
    in Hanswer.
  set (stenv := Meet_env.typing_env env4) in *.
  destruct (row_join_result_env need_join stenv (Meet_env.names_to_avoid env4)
              state)
    as [ state1 | ] eqn:Hpush; [ | discriminate Hanswer ].
  cbv beta iota zeta in Hanswer. rewrite Hlift in Hanswer.
  destruct (row_push env4 state state1 Hinv Hextends Hfits4 Hnames Hpush)
    as (Hinv1 & Hle1 & Hin1 & _).
  match type of Hanswer with
  | context [row_add_extra _ _ ?s] =>
    remember s as state2 eqn:Hstate2 in Hanswer
  end.
  assert (Hsame2 : same_but_flags state1 state2).
  { rewrite Hstate2.
    exact (same_but_flags_trans _ _ _ (update_refs_same ri state1)
             (update_refs_same rm _)). }
  clear Hstate2.
  assert (Hinv2 := same_but_flags_inv _ _ Hsame2 Hinv1).
  assert (Hle2 : state_le state state2).
  { exact (state_le_trans _ _ _ Hle1 (proj1 (same_but_flags_le _ _ Hsame2))). }
  assert (Hin2 : stenv ∈ envs_of state2).
  { rewrite (same_but_flags_envs _ _ Hsame2). exact Hin1. }
  assert (Hscoped : tenv_extends (Typing_env.increment_scope tenv) stenv).
  { exact (meet_env_extends_tenv _ _ Hextends). }
  (* What is answered, whatever the extension. *)
  assert (Hfinal :
    ∀ (extension : env_extension) (state3 : row_like_state),
      row_inv state3 ->
      state_le state state3 ->
      stenv ∈ envs_of state3 ->
      covered stenv (extra_variables state3) ->
      extension_fits stenv extension ->
      (∀ (statics : static_env) (valuation4 : env),
         typing_env_satisfied_strictly statics valuation4 stenv ->
         equations_valued stenv (Typing_env.current_scope tenv) valuation4 ->
         env_extension_satisfied statics valuation4 extension) ->
      row_inv state3
      ∧ state_le state state3
      ∧ result_placed o
          (Some (Or_unknown.Known (Row_like_case maps' index' extension)))
          state3
      ∧ pair_sound (Some (Or_unknown.Known c1)) (Some (Or_unknown.Known c2))
          (Some (Or_unknown.Known (Row_like_case maps' index' extension)))
          state3).
  { intros extension state3 Hinv3 Hle3 Hin3 Hcovered Hextension Hextension_true.
    assert (Hcase : case_fits stenv o (Row_like_case maps' index' extension)).
    { split; [ exact Hindex | ]. split; [ exact Hmaps | exact Hextension ]. }
    split; [ exact Hinv3 | ]. split; [ exact Hle3 | ]. split.
    { right. exists stenv. split; [ exact Hin3 | ].
      split; [ exact Hcovered | exact Hcase ]. }
    intros statics valuation Hstatics Hsatisfied Htrue1 Htrue2.
    destruct (Hlost statics valuation Hstatics Hsatisfied Htrue1 Htrue2)
      as (valuation4 & Hextended4 & Hsatisfied4 & Hvalued4 & Hindex_true4
          & Hholds4).
    exists stenv, valuation4. split; [ exact Hin3 | ].
    split; [ exact Hextended4 | ]. split; [ exact Hsatisfied4 | ].
    split; [ exact Hvalued4 | ].
    intros valuation' Hextended'.
    apply (case_true_stable statics stenv valuation4 valuation' o _ Hcase
             Hextended').
    split; [ exact Hindex_true4 | ].
    split; [ exact (Htrue statics valuation4 maps' Hholds4) | ].
    apply (Hextension_true statics valuation4 Hsatisfied4).
    exact (Hvalued4 _ (increment_scope_valued tenv valuation (proj1 Hfits))). }
  destruct need_join; cbv beta iota in Hanswer.
  - destruct (row_add_extra common_scope stenv state2)
      as [ [extension state3] | ] eqn:Hextra; [ | discriminate Hanswer ].
    cbv beta iota in Hanswer.
    destruct (row_add_extra_spec stenv state2 state3 extension Hinv2 Hin2
                Hextra)
      as (-> & Hinv3 & Hle3 & Henvs3 & Hcovered).
    assert (Hin3 : stenv ∈ envs_of state3) by (rewrite Henvs3; exact Hin2).
    assert (Hresult :=
      Hfinal (extension_of_level (level_of stenv)) state3 Hinv3
        (state_le_trans _ _ _ Hle2 Hle3) Hin3 Hcovered
        (scoped_extension_fits tenv stenv Hfits4 Hscoped)
        (scoped_extension_satisfied tenv stenv Hfits4 Hscoped)).
    destruct (extension_of_level (level_of stenv)) as [ [ | binding rest ] ]
      eqn:Hlevel; injection Hanswer as <- <-.
    + exact Hresult.
    + exact (row_step_flags o _ _ _ state state3 _
               (same_but_flags_trans _ _ _ (not_t2_same state3)
                  (not_t1_same _))
               Hresult).
  - destruct (row_add_extra common_scope stenv state2)
      as [ [extension state3] | ] eqn:Hextra; [ | discriminate Hanswer ].
    cbv beta iota in Hanswer. injection Hanswer as <- <-.
    destruct (row_add_extra_spec stenv state2 state3 extension Hinv2 Hin2
                Hextra)
      as (_ & Hinv3 & Hle3 & Henvs3 & Hcovered).
    assert (Hin3 : stenv ∈ envs_of state3) by (rewrite Henvs3; exact Hin2).
    apply (Hfinal _ state3 Hinv3 (state_le_trans _ _ _ Hle2 Hle3) Hin3 Hcovered
             (empty_extension_fits stenv)).
    intros statics valuation4 _ _. apply empty_extension_satisfied.
Qed.

(* Two cases are met. *)
Lemma row_meet_case_step (o : option row_tag) (c1 c2 : case)
    (state : row_like_state) :
  row_inv state ->
  case_fits tenv o c1 ->
  case_fits tenv o c2 ->
  row_step o (Some (Or_unknown.Known c1)) (Some (Or_unknown.Known c2)) state
    (row_meet_case common_scope need_join base c1 c2 state).
Proof.
  intros Hinv (Hix1 & Hm1 & He1) (Hix2 & Hm2 & He2) r state' Hanswer.
  unfold row_meet_case in Hanswer.
  set (env0 := Meet_env.with_names_to_avoid base (avoid state)) in *.
  (* A step that answers no case, and is right because the two cases
     have no value in common. *)
  assert (Hnone :
    ∀ state1 : row_like_state,
      same_but_flags state state1 ->
      (∀ (statics : static_env) (valuation : env),
         statics_valid statics ->
         typing_env_satisfied_strictly statics valuation tenv ->
         case_true statics valuation c1 ->
         case_true statics valuation c2 ->
         False) ->
      row_inv state1
      ∧ state_le state state1
      ∧ result_placed o None state1
      ∧ pair_sound (Some (Or_unknown.Known c1)) (Some (Or_unknown.Known c2))
          None state1).
  { intros state1 Hsame Hfalse.
    apply (row_step_flags o _ _ None state state state1 Hsame).
    split; [ exact Hinv | ]. split; [ apply state_le_refl | ].
    split; [ exact I | ].
    intros statics valuation Hstatics Hsatisfied H1 H2.
    destruct (Hfalse statics valuation Hstatics Hsatisfied H1 H2). }
  destruct (row_meet_index env0 (case_index c1) (case_index c2))
    as [ ri0 | ri env1 ] eqn:Hindex.
  { injection Hanswer as <- <-. apply (Hnone _ (update_refs_same ri0 state)).
    intros statics valuation _ _ (Hi1 & _) (Hi2 & _).
    destruct (Hindex_true o env0 _ _ _ _ Hix1 Hix2 Hi1 Hi2) as (ri & Heq & _).
    rewrite Hindex in Heq. discriminate Heq. }
  assert (Henv1 := row_meet_index_env _ _ _ _ _ Hindex). subst env1.
  assert (Hm1' := Hmaps_kinds tenv _ _ (base_knows (avoid state)) Hm1).
  assert (Hm2' := Hmaps_kinds tenv _ _ (base_knows (avoid state)) Hm2).
  destruct (meet_maps_to env0 (case_maps_to c1) (case_maps_to c2))
    as [ maps_result | ] eqn:Hmaps; [ | discriminate Hanswer ].
  destruct (Hmeet_maps env0 _ _ maps_result (base_fits _) Hm1' Hm2' Hmaps)
    as [Hmaps_keeps Hmaps_sound].
  (* No value is lost by the meet of the two products. *)
  assert (Hlost2 :
    ∀ (statics : static_env) (valuation : env),
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation tenv ->
      case_true statics valuation c1 ->
      case_true statics valuation c2 ->
      ∃ (rm : meet_return_value maps_to) (env2 : Meet_env.t)
        (valuation2 : env),
        maps_result = Meet_result.Ok rm env2
        ∧ extended_from tenv valuation valuation2
        ∧ typing_env_satisfied_strictly statics valuation2
            (Meet_env.typing_env env2)
        ∧ maps_holds statics valuation2
            (extract_value rm (case_maps_to c1) (case_maps_to c2))
        ∧ keeps_equations_valued (Typing_env.increment_scope tenv)
            (Meet_env.typing_env env2) valuation valuation2).
  { intros statics valuation Hstatics Hsatisfied (Hi1 & Ht1 & _)
      (Hi2 & Ht2 & _).
    exact (Hmaps_sound statics valuation Hstatics
             (base_satisfied _ statics valuation Hsatisfied)
             (Hholds statics valuation o _ _ Hstatics Hix1 Hi1 Ht1)
             (Hholds statics valuation o _ _ Hstatics Hix2 Hi2 Ht2)). }
  destruct maps_result as [ rm0 | rm env2 ].
  { injection Hanswer as <- <-. apply (Hnone _ (update_refs_same rm0 state)).
    intros statics valuation Hstatics Hsatisfied H1 H2.
    destruct (Hlost2 statics valuation Hstatics Hsatisfied H1 H2)
      as (rm & env2 & valuation2 & Hbottom & _).
    discriminate Hbottom. }
  destruct (Hmaps_keeps rm env2 eq_refl)
    as (Hextends2 & Hfits2 & Hanswer_m & Hnames2).
  assert (Hknows2 :
    knows_at_kinds tenv (Meet_env.typing_env env2)).
  { exact (knows_at_kinds_trans _ _ _ (base_knows (avoid state))
             (meet_env_extends_knows _ _ Hextends2)). }
  assert (Hext1 :
    optional_extension_fits (Meet_env.typing_env env2)
      (Some (case_env_extension c1))).
  { exact (extension_fits_kept tenv _ _ Hknows2 He1). }
  (* The first extension is added and no value is lost. *)
  assert (Hlost3 :
    ∀ (statics : static_env) (valuation : env)
      (added : Or_bottom.t Meet_env.t),
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation tenv ->
      case_true statics valuation c1 ->
      case_true statics valuation c2 ->
      Meet_env.add_env_extension_strict env2 (case_env_extension c1) mh
      = Some added ->
      ∃ (env3 : Meet_env.t) (valuation3 : env),
        added = Or_bottom.Ok env3
        ∧ extended_from tenv valuation valuation3
        ∧ extended_from (Meet_env.typing_env env2) valuation3 valuation3
        ∧ typing_env_satisfied_strictly statics valuation3
            (Meet_env.typing_env env3)
        ∧ (∀ valuation' : env,
             extended_from (Meet_env.typing_env env2) valuation3 valuation' ->
             maps_holds statics valuation'
               (extract_value rm (case_maps_to c1) (case_maps_to c2)))
        ∧ keeps_equations_valued (Typing_env.increment_scope tenv)
            (Meet_env.typing_env env3) valuation valuation3).
  { intros statics valuation added Hstatics Hsatisfied Htrue1 Htrue2 Hadd.
    destruct (Hlost2 statics valuation Hstatics Hsatisfied Htrue1 Htrue2)
      as (rm' & env2' & valuation2 & [= <- <-] & Hextended2 & Hsatisfied2
          & Hholds2 & Hvalued2).
    destruct (add_optional_extension_sound mh Hkeeps Hsound statics valuation2
                env2 _ added Hfits2 Hext1 Hstatics Hsatisfied2
                (env_extension_satisfied_extended statics tenv valuation
                   valuation2 _ (proj1 (proj2 He1)) Hextended2
                   (proj2 (proj2 Htrue1)))
                Hadd)
      as (env3 & -> & valuation3 & Hextended3 & Hsatisfied3 & Hvalued3).
    exists env3, valuation3. split; [ reflexivity | ]. split.
    { intros x Hx. rewrite (Hextended3 x (proj1 Hknows2 _ Hx)).
      exact (Hextended2 x Hx). }
    split; [ apply extended_from_refl | ]. split; [ exact Hsatisfied3 | ].
    split.
    - intros valuation' Hextended'.
      apply (proj2 Hmaps_stable statics (Meet_env.typing_env env2) valuation2
               valuation' _ Hanswer_m); [ | exact Hholds2 ].
      intros x Hx. rewrite (Hextended' x Hx). exact (Hextended3 x Hx).
    - exact (keeps_equations_valued_trans _ _ _ _ _ _ Hvalued2 Hvalued3). }
  destruct (Meet_env.add_env_extension_strict env2 (case_env_extension c1) mh)
    as [ [ env3 | ] | ] eqn:Hadd1; [ | | discriminate Hanswer ].
  2:{ cbv beta iota in Hanswer. injection Hanswer as <- <-.
      apply (Hnone _ (update_refs_same (@New_result unit tt) state)).
      intros statics valuation Hstatics Hsatisfied H1 H2.
      destruct (Hlost3 statics valuation _ Hstatics Hsatisfied H1 H2 eq_refl)
        as (env3 & valuation3 & Hbottom & _).
      discriminate Hbottom. }
  destruct (add_optional_extension_keeps mh Hkeeps env2 env3 _ Hfits2 Hext1
              Hadd1)
    as [[Hextends3 Hfits3] Hnames3].
  assert (Hknows3 : knows_at_kinds tenv (Meet_env.typing_env env3)).
  { exact (knows_at_kinds_trans _ _ _ Hknows2
             (meet_env_extends_knows _ _ Hextends3)). }
  assert (Hext2 :
    optional_extension_fits (Meet_env.typing_env env3)
      (Some (case_env_extension c2))).
  { exact (extension_fits_kept tenv _ _ Hknows3 He2). }
  (* The second extension is added and no value is lost. *)
  assert (Hlost4 :
    ∀ (statics : static_env) (valuation : env)
      (added : Or_bottom.t Meet_env.t),
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation tenv ->
      case_true statics valuation c1 ->
      case_true statics valuation c2 ->
      Meet_env.add_env_extension_strict env3 (case_env_extension c2) mh
      = Some added ->
      ∃ (env4 : Meet_env.t) (valuation4 : env),
        added = Or_bottom.Ok env4
        ∧ extended_from tenv valuation valuation4
        ∧ typing_env_satisfied_strictly statics valuation4
            (Meet_env.typing_env env4)
        ∧ maps_holds statics valuation4
            (extract_value rm (case_maps_to c1) (case_maps_to c2))
        ∧ keeps_equations_valued (Typing_env.increment_scope tenv)
            (Meet_env.typing_env env4) valuation valuation4).
  { intros statics valuation added Hstatics Hsatisfied Htrue1 Htrue2 Hadd.
    destruct (Hlost3 statics valuation _ Hstatics Hsatisfied Htrue1 Htrue2
                eq_refl)
      as (env3' & valuation3 & [= <-] & Hextended3 & _ & Hsatisfied3 & Hholds3
          & Hvalued3).
    destruct (add_optional_extension_sound mh Hkeeps Hsound statics valuation3
                env3 _ added Hfits3 Hext2 Hstatics Hsatisfied3
                (env_extension_satisfied_extended statics tenv valuation
                   valuation3 _ (proj1 (proj2 He2)) Hextended3
                   (proj2 (proj2 Htrue2)))
                Hadd)
      as (env4 & -> & valuation4 & Hextended4 & Hsatisfied4 & Hvalued4).
    exists env4, valuation4. split; [ reflexivity | ]. split.
    { intros x Hx. rewrite (Hextended4 x (proj1 Hknows3 _ Hx)).
      exact (Hextended3 x Hx). }
    split; [ exact Hsatisfied4 | ]. split.
    - apply Hholds3. intros x Hx.
      exact (Hextended4 x
               (proj1 (meet_env_extends_knows _ _ Hextends3) _ Hx)).
    - exact (keeps_equations_valued_trans _ _ _ _ _ _ Hvalued3 Hvalued4). }
  cbv beta iota in Hanswer.
  destruct (Meet_env.add_env_extension_strict env3 (case_env_extension c2) mh)
    as [ [ env4 | ] | ] eqn:Hadd2; [ | | discriminate Hanswer ].
  2:{ cbv beta iota in Hanswer. injection Hanswer as <- <-.
      apply (Hnone _ (update_refs_same (@New_result unit tt) state)).
      intros statics valuation Hstatics Hsatisfied H1 H2.
      destruct (Hlost4 statics valuation _ Hstatics Hsatisfied H1 H2 eq_refl)
        as (env4 & valuation4 & Hbottom & _).
      discriminate Hbottom. }
  destruct (add_optional_extension_keeps mh Hkeeps env3 env4 _ Hfits3 Hext2
              Hadd2)
    as [[Hextends4 Hfits4] Hnames4].
  cbv beta iota delta [Meet_env.final_typing_env_strict] in Hanswer.
  assert (Hextends24 := meet_env_extends_trans _ _ _ Hextends3 Hextends4).
  refine (row_case_result_step o c1 c2 ri rm env4 state Hinv
            (meet_env_extends_trans _ _ _ Hextends2 Hextends24) Hfits4
            (new_names_avoided_trans _ _ _ Hextends2 Hextends24 Hnames2
               (new_names_avoided_trans _ _ _ Hextends3 Hextends4 Hnames3
                  Hnames4))
            (Hindex_ok o env0 env0 env2 _ _ _ _ ri rm Hix1 Hix2 Hindex Hmaps)
            (proj1 Hmaps_stable env2 env4 _ Hextends24 Hanswer_m) _ r state'
            Hanswer).
  intros statics valuation Hstatics Hsatisfied Htrue1 Htrue2.
  destruct (Hlost4 statics valuation _ Hstatics Hsatisfied Htrue1 Htrue2
              eq_refl)
    as (env4' & valuation4 & [= <-] & Hextended4 & Hsatisfied4 & Hholds4
        & Hvalued4).
  exists valuation4. split; [ exact Hextended4 | ].
  split; [ exact Hsatisfied4 | ]. split; [ exact Hvalued4 | ].
  split; [ | exact Hholds4 ].
  destruct (Hindex_true o env0 _ _ _ _ Hix1 Hix2 (proj1 Htrue1) (proj1 Htrue2))
    as (ri' & Heq & Hindex_true').
  rewrite Hindex in Heq. injection Heq as <-. exact Hindex_true'.
Qed.

Lemma case_fits_other (stenv : Typing_env.t) (tag : row_tag) (c : case) :
  case_fits stenv None c -> case_fits stenv (Some tag) c.
Proof.
  intros (Hindex & Hrest). split; [ | exact Hrest ].
  exact (Hindex_other tag _ _ Hindex).
Qed.

(* The cases of a tag. *)
Lemma row_meet_knowns_step (tag : row_tag) (other1 other2 : Or_bottom.t case)
    (c1 c2 : option (Or_unknown.t case)) (state : row_like_state) :
  row_inv state ->
  sel_fits tenv (Some tag) c1 ->
  sel_fits tenv (Some tag) c2 ->
  (∀ c : case, other1 = Or_bottom.Ok c -> case_fits tenv None c) ->
  (∀ c : case, other2 = Or_bottom.Ok c -> case_fits tenv None c) ->
  ∀ (r : option (Or_unknown.t case)) (state' : row_like_state),
    row_meet_knowns common_scope need_join base other1 other2 c1 c2 state
    = Some (r, state') ->
    row_inv state'
    ∧ state_le state state'
    ∧ result_placed (Some tag) r state'
    ∧ (c1 <> None ∨ c2 <> None ->
       pair_sound (selected c1 other1) (selected c2 other2) r state').
Proof.
  intros Hinv Hc1 Hc2 Hother1 Hother2 r state' Hanswer.
  assert (Hweaken :
    ∀ s1 s2 : option (Or_unknown.t case),
      row_inv state'
      ∧ state_le state state'
      ∧ result_placed (Some tag) r state'
      ∧ pair_sound s1 s2 r state' ->
      s1 = selected c1 other1 ->
      s2 = selected c2 other2 ->
      row_inv state'
      ∧ state_le state state'
      ∧ result_placed (Some tag) r state'
      ∧ (c1 <> None ∨ c2 <> None ->
         pair_sound (selected c1 other1) (selected c2 other2) r state')).
  { intros s1 s2 (H1 & H2 & H3 & H4) -> ->. auto. }
  assert (Hnone :
    ∀ state1 : row_like_state,
      same_but_flags state state1 ->
      (selected c1 other1 = None ∨ selected c2 other2 = None) ->
      r = None ->
      state' = state1 ->
      row_inv state'
      ∧ state_le state state'
      ∧ result_placed (Some tag) r state'
      ∧ (c1 <> None ∨ c2 <> None ->
         pair_sound (selected c1 other1) (selected c2 other2) r state')).
  { intros state1 Hsame Hno -> ->.
    split; [ exact (same_but_flags_inv _ _ Hsame Hinv) | ].
    split; [ exact (proj1 (same_but_flags_le _ _ Hsame)) | ].
    split; [ exact I | ].
    intros _ statics valuation _ _ H1 H2.
    destruct Hno as [ Hno | Hno ]; rewrite Hno in *;
      [ destruct H1 | destruct H2 ]. }
  unfold row_meet_knowns in Hanswer.
  destruct c1 as [ [ c1 | ] | ], c2 as [ [ c2 | ] | ].
  - exact (Hweaken _ _
             (row_meet_case_step (Some tag) c1 c2 state Hinv Hc1 Hc2 r state'
                Hanswer)
             eq_refl eq_refl).
  - refine (Hweaken _ _
              (row_with_unknown_step (Some tag) c1 not_t2
                 (Some (Or_unknown.Known c1)) (Some Or_unknown.Unknown) state
                 Hinv Hc1 not_t2_same _ r state' Hanswer)
              eq_refl eq_refl).
    intros statics valuation H1 _. exact H1.
  - destruct other2 as [ oc | ].
    + exact (Hweaken _ _
               (row_meet_case_step (Some tag) c1 oc state Hinv Hc1
                  (case_fits_other tenv tag oc (Hother2 oc eq_refl)) r state'
                  Hanswer)
               eq_refl eq_refl).
    + injection Hanswer as <- <-.
      apply (Hnone _ (not_t1_same state)); auto.
  - refine (Hweaken _ _
              (row_with_unknown_step (Some tag) c2 not_t1
                 (Some Or_unknown.Unknown) (Some (Or_unknown.Known c2)) state
                 Hinv Hc2 not_t1_same _ r state' Hanswer)
              eq_refl eq_refl).
    intros statics valuation _ H2. exact H2.
  - exact (Hweaken _ _
             (row_unknowns_step (Some tag) state Hinv r state' Hanswer)
             eq_refl eq_refl).
  - destruct other2 as [ oc | ].
    + refine (Hweaken _ _
                (row_with_unknown_step (Some tag) oc
                   (fun state => not_t1 (not_t2 state))
                   (Some Or_unknown.Unknown) (Some (Or_unknown.Known oc)) state
                   Hinv (case_fits_other tenv tag oc (Hother2 oc eq_refl)) _ _
                   r state' Hanswer)
                eq_refl eq_refl).
      * intros s.
        exact (same_but_flags_trans _ _ _ (not_t2_same s) (not_t1_same _)).
      * intros statics valuation _ H2. exact H2.
    + injection Hanswer as <- <-.
      apply (Hnone _ (not_t1_same state)); auto.
  - destruct other1 as [ oc | ].
    + exact (Hweaken _ _
               (row_meet_case_step (Some tag) oc c2 state Hinv
                  (case_fits_other tenv tag oc (Hother1 oc eq_refl)) Hc2 r
                  state' Hanswer)
               eq_refl eq_refl).
    + injection Hanswer as <- <-.
      apply (Hnone _ (not_t2_same state)); auto.
  - destruct other1 as [ oc | ].
    + refine (Hweaken _ _
                (row_with_unknown_step (Some tag) oc
                   (fun state => not_t1 (not_t2 state))
                   (Some (Or_unknown.Known oc)) (Some Or_unknown.Unknown) state
                   Hinv (case_fits_other tenv tag oc (Hother1 oc eq_refl)) _ _
                   r state' Hanswer)
                eq_refl eq_refl).
      * intros s.
        exact (same_but_flags_trans _ _ _ (not_t2_same s) (not_t1_same _)).
      * intros statics valuation H1 _. exact H1.
    + injection Hanswer as <- <-.
      apply (Hnone _ (not_t2_same state)); auto.
  - injection Hanswer as <- <-.
    split; [ exact Hinv | ]. split; [ apply state_le_refl | ].
    split; [ exact I | ]. intros [ H | H ]; destruct H; reflexivity.
Qed.

(* The cases of the other tags. *)
Lemma row_meet_others_step (other1 other2 other : Or_bottom.t case)
    (state state' : row_like_state) :
  row_inv state ->
  (∀ c : case, other1 = Or_bottom.Ok c -> case_fits tenv None c) ->
  (∀ c : case, other2 = Or_bottom.Ok c -> case_fits tenv None c) ->
  row_meet_others common_scope need_join base other1 other2 state
  = Some (other, state') ->
  row_inv state'
  ∧ state_le state state'
  ∧ (∀ c : case, other = Or_bottom.Ok c -> case_placed None c state')
  ∧ pair_sound (selected None other1) (selected None other2)
      (selected None other) state'.
Proof.
  intros Hinv Hother1 Hother2 Hanswer.
  assert (Hnone :
    ∀ state1 : row_like_state,
      same_but_flags state state1 ->
      (selected None other1 = None ∨ selected None other2 = None) ->
      other = Or_bottom.Bottom ->
      state' = state1 ->
      row_inv state'
      ∧ state_le state state'
      ∧ (∀ c : case, other = Or_bottom.Ok c -> case_placed None c state')
      ∧ pair_sound (selected None other1) (selected None other2)
          (selected None other) state').
  { intros state1 Hsame Hno -> ->.
    split; [ exact (same_but_flags_inv _ _ Hsame Hinv) | ].
    split; [ exact (proj1 (same_but_flags_le _ _ Hsame)) | ].
    split; [ intros c Hc; discriminate Hc | ].
    intros statics valuation _ _ H1 H2.
    destruct Hno as [ Hno | Hno ]; rewrite Hno in *;
      [ destruct H1 | destruct H2 ]. }
  unfold row_meet_others in Hanswer.
  destruct other1 as [ c1 | ], other2 as [ c2 | ].
  - destruct (row_meet_case common_scope need_join base c1 c2 state)
      as [ [r state1] | ] eqn:Hcase; [ | discriminate Hanswer ].
    destruct (row_meet_case_step None c1 c2 state Hinv (Hother1 c1 eq_refl)
                (Hother2 c2 eq_refl) r state1 Hcase)
      as (Hinv1 & Hle1 & Hplaced & Hpair).
    cbv beta iota in Hanswer.
    destruct r as [ [ c | ] | ]; [ | discriminate Hanswer | ];
      injection Hanswer as <- <-.
    + split; [ exact Hinv1 | ]. split; [ exact Hle1 | ].
      split; [ intros c' [= <-]; exact Hplaced | exact Hpair ].
    + split; [ exact Hinv1 | ]. split; [ exact Hle1 | ].
      split; [ intros c' Hc'; discriminate Hc' | exact Hpair ].
  - injection Hanswer as <- <-.
    apply (Hnone _ (not_t1_same state)); auto.
  - injection Hanswer as <- <-.
    apply (Hnone _ (not_t2_same state)); auto.
  - injection Hanswer as <- <-. split; [ exact Hinv | ].
    split; [ apply state_le_refl | ].
    split; [ intros c Hc; discriminate Hc | ].
    intros statics valuation _ _ [].
Qed.

(* ======================================================================== *)
(* The environments kept are joined                                         *)
(* ======================================================================== *)

(* What an environment knows after variables have been defined in it,
   if it knew what the environment of the meet knows, had what it knew
   of an environment kept at the kinds of that one, and the variables
   defined cover the level of that one: all that the environment kept
   knows, at its kinds. *)
Lemma covered_knows_after (stenv : Typing_env.t)
    (kind_of : variable -> option flambda_kind) (vars : list variable)
    (menv0 menv1 : Meet_env.t) :
  typing_env_fits stenv ->
  tenv_extends (Typing_env.increment_scope tenv) stenv ->
  knows_at_kinds tenv (Meet_env.typing_env menv0) ->
  (∀ (s : simple) (kind : flambda_kind),
     Typing_env.mem_simple None (Meet_env.typing_env menv0) s = true ->
     simple_of_kind stenv s kind ->
     simple_of_kind (Meet_env.typing_env menv0) s kind) ->
  (∀ (y : variable) (k : flambda_kind),
     Typing_env_level.find_kind (level_of stenv) y = Some k ->
     y ∈ vars ∧ kind_of y = Some k) ->
  defined_from kind_of vars menv0 menv1 ->
  knows_at_kinds stenv (Meet_env.typing_env menv1).
Proof.
  intros Hscoped_fits Hscoped [Hmem0 Hkind0] Hagree Hcover
    (Hextends & _ & _ & _ & Hkind & Hknown & _).
  assert (Htenv := meet_env_extends_tenv _ _ Hextends).
  assert (Hnew :
    ∀ x : variable,
      Typing_env.mem None stenv (Name_var x) = true ->
      Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x) = false ->
      ∃ k : flambda_kind,
        Typing_env_level.find_kind (level_of stenv) x = Some k).
  { intros x Hx Hunknown.
    apply (scoped_new_in_order tenv stenv Hfits Hscoped_fits Hscoped x Hx).
    destruct (Typing_env.mem None tenv (Name_var x)) eqn:Hold;
      [ | reflexivity ].
    rewrite (Hmem0 _ Hold) in Hunknown. discriminate Hunknown. }
  split.
  - intros n Hn. destruct n as [ x | sym ].
    + destruct (Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x))
        eqn:Hx.
      * exact (tenv_extends_mem _ _ _ Htenv Hx).
      * destruct (Hnew x Hn Hx) as [k Hk].
        destruct (Hcover x k Hk) as [Hin Hkind_of].
        exact (Hknown x k Hin Hkind_of).
    + apply (tenv_extends_mem _ _ _ Htenv). apply Hmem0.
      rewrite <- (scoped_symbol_known tenv stenv Hscoped). exact Hn.
  - intros s kind Hs.
    destruct (Typing_env.mem_simple None (Meet_env.typing_env menv0) s)
      eqn:Hknown0.
    + exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv)))) s kind
               (Hagree s kind Hknown0 Hs)).
    + destruct s as [ x | sym | c ]; [ | | discriminate Hknown0 ].
      * destruct (Hnew x
                    (simple_of_kind_mem stenv (Name_var x) kind
                       (proj1 Hscoped_fits) Hs)
                    Hknown0) as [k Hk].
        assert (Hsame : k = kind).
        { exact (simple_of_kind_unique stenv (Var x) k kind
                   (scoped_defined_kind tenv stenv Hscoped_fits Hscoped x k Hk)
                   Hs). }
        subst k. destruct (Hcover x kind Hk) as [Hin Hkind_of].
        exact (Hkind x kind Hin Hkind_of Hknown0).
      * exfalso.
        assert (Hsym := simple_of_kind_mem stenv (Name_symbol sym) kind
                          (proj1 Hscoped_fits) Hs).
        rewrite (scoped_symbol_known tenv stenv Hscoped) in Hsym.
        cbn [Typing_env.mem_simple] in Hknown0.
        rewrite (Hmem0 _ Hsym) in Hknown0. discriminate Hknown0.
Qed.

(* A variable is known to one of the environments of a list, or to
   none. *)
Lemma known_in_some (l : list Typing_env.t) (x : variable) :
  (∃ e : Typing_env.t,
     e ∈ l ∧ Typing_env.mem None e (Name_var x) = true)
  ∨ (∀ e : Typing_env.t,
       e ∈ l -> Typing_env.mem None e (Name_var x) = false).
Proof.
  induction l as [ | a l IH ]; [ right; intros e [] | ].
  destruct (Typing_env.mem None a (Name_var x)) eqn:Ha.
  - left. exists a. split; [ left; reflexivity | exact Ha ].
  - destruct IH as [ (e & Hin & He) | Hnone ].
    + left. exists e. split; [ right; exact Hin | exact He ].
    + right. intros e [ <- | Hin ]; [ exact Ha | exact (Hnone e Hin) ].
Qed.

(* The environment of the meet, with more names to avoid. *)
Lemma avoiding_extends (names : list variable) :
  (∀ x : variable,
     among x (Meet_env.names_to_avoid menv) = true -> among x names = true) ->
  meet_env_extends menv (Meet_env.with_names_to_avoid menv names).
Proof.
  intros Hnames. repeat split; try reflexivity.
  - intros s kind Hs. exact Hs.
  - exact Hnames.
  - apply keeps_names_refl.
Qed.

(* What the join of the environments kept and the definition of the
   extra variables give. *)
Lemma row_join_all_spec (state : row_like_state)
    (scoped_envs : list Typing_env.t) (env_final : Meet_env.t) :
  row_inv state ->
  envs_of state = scoped_envs ->
  row_join_all menv common_scope tenv scoped_envs state = Some env_final ->
  meet_env_extends menv env_final
  ∧ typing_env_fits (Meet_env.typing_env env_final)
  ∧ new_names_avoided menv env_final
  ∧ (∀ stenv : Typing_env.t,
       stenv ∈ scoped_envs ->
       covered stenv (extra_variables state) ->
       knows_at_kinds stenv (Meet_env.typing_env env_final))
  ∧ ∀ (stenv : Typing_env.t) (statics : static_env)
      (valuation valuation_k : env),
      stenv ∈ scoped_envs ->
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation tenv ->
      extended_from tenv valuation valuation_k ->
      typing_env_satisfied_strictly statics valuation_k stenv ->
      keeps_equations_valued (Typing_env.increment_scope tenv) stenv valuation
        valuation_k ->
      ∃ valuation' : env,
        extended_from stenv valuation_k valuation'
        ∧ typing_env_satisfied_strictly statics valuation'
            (Meet_env.typing_env env_final)
        ∧ keeps_equations_valued tenv (Meet_env.typing_env env_final) valuation
            valuation'.
Proof.
  intros [Hscoped Hapart Hnew Havoid Hone Hextra] Henvs Hall.
  rewrite Henvs in *. unfold row_join_all in Hall.
  set (joining := Meet_env.with_names_to_avoid menv (avoid state)) in *.
  destruct (join mh common_scope joining tenv scoped_envs)
    as [ joined | ] eqn:Hjoined; [ | discriminate Hall ].
  cbv beta iota in Hall.
  set (extra := extra_variables state) in *.
  set (kind_of := fun x : variable => lookup x extra).
  rewrite (define_extra_all choices kind_of extra Hone) in Hall.
  assert (Hscoped_of :
    ∀ scoped_env : Typing_env.t,
      scoped_env ∈ scoped_envs -> scoped_env_of tenv scoped_env).
  { intros e Hin. destruct (Hscoped e Hin) as [Hextends_e Hfits_e].
    exact (scoped_scoped_env_of tenv e Hfits_e Hextends_e). }
  assert (Hagree : scoped_envs_agree scoped_envs).
  { intros e e' s kind kind' Hin Hin' Hs Hs'.
    destruct (Hscoped e Hin) as [Hextends_e Hfits_e].
    destruct (Hscoped e' Hin') as [Hextends_e' Hfits_e'].
    destruct (envs_apart_in _ Hapart e e' Hin Hin') as [ <- | Hdisjoint ].
    { exact (simple_of_kind_unique e s kind kind' Hs Hs'). }
    destruct (Typing_env.mem_simple None tenv s) eqn:Hknown.
    - exact (simple_of_kind_unique tenv s kind kind'
               (scoped_env_of_kind tenv e s kind (Hscoped_of e Hin) Hknown Hs)
               (scoped_env_of_kind tenv e' s kind' (Hscoped_of e' Hin') Hknown
                  Hs')).
    - exfalso. destruct s as [ x | sym | c ]; [ | | discriminate Hknown ].
      + cbn [Typing_env.mem_simple] in Hknown.
        rewrite (Hdisjoint x
                   (simple_of_kind_mem e (Name_var x) kind (proj1 Hfits_e) Hs)
                   (simple_of_kind_mem e' (Name_var x) kind' (proj1 Hfits_e')
                      Hs'))
          in Hknown.
        discriminate Hknown.
      + cbn [Typing_env.mem_simple] in Hknown.
        rewrite <- (scoped_symbol_known tenv e Hextends_e sym) in Hknown.
        rewrite (simple_of_kind_mem e (Name_symbol sym) kind (proj1 Hfits_e)
                   Hs)
          in Hknown.
        discriminate Hknown. }
  destruct (Hjoin mh joining joined scoped_envs Hsound Hkeeps Hfits Hscoped_of
              Hagree Hjoined)
    as (Jsound & Jextends & Jfits & Jnew & Jterms & Jkinds).
  assert (Hdefined :=
    define_all_defined choices Hdefine kind_of (map fst extra) joined
      env_final Jfits Hall).
  assert (Hdefined' := Hdefined).
  destruct Hdefined'
    as (Dextends & Dfits & Dnames & Dmem & Dkind & Dknown & Dsatisfied
        & Dvalued).
  assert (Hextends0 : meet_env_extends menv joined).
  { exact (meet_env_extends_trans _ _ _ (avoiding_extends _ Havoid) Jextends). }
  assert (Hknows0 : knows_at_kinds tenv (Meet_env.typing_env joined)).
  { exact (meet_env_extends_knows _ _ Hextends0). }
  (* A variable among the extra variables is one that the level of an
     environment kept defines. *)
  assert (Hvars :
    ∀ (y : variable) (k : flambda_kind),
      kind_of y = Some k ->
      ∃ stenv : Typing_env.t,
        stenv ∈ scoped_envs
        ∧ Typing_env_level.find_kind (level_of stenv) y = Some k).
  { intros y k Hk. exact (Hextra y k Hk). }
  split; [ exact (meet_env_extends_trans _ _ _ Hextends0 Dextends) | ].
  split; [ exact Dfits | ]. split; [ | split ].
  - (* The variables that are new. They may not occur in terms: not
       in the result of the join, and the extra variables are defined
       after it in the name mode of types. *)
    intros x Hknown Hunknown. rewrite Dnames.
    assert (Hterms :
      Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env env_final)
        (Name_var x)
      = false).
    { apply (define_all_not_in_terms choices kind_of (map fst extra) x joined
               env_final Hall).
      destruct (Typing_env.mem None (Meet_env.typing_env joined) (Name_var x))
        eqn:Hjoined_x;
        [ exact (Jterms x Hjoined_x Hunknown)
        | exact (unknown_not_in_terms _ x Hjoined_x) ]. }
    enough (Htwo :
      among x (Meet_env.names_to_avoid menv) = false
      ∧ among x (Meet_env.names_to_avoid joined) = true)
      by (destruct Htwo as [Hfresh Hamong];
          split; [ exact Hfresh | split; [ exact Hamong | exact Hterms ] ]).
    assert (Hsome :
      ∀ e : Typing_env.t,
        e ∈ scoped_envs ->
        Typing_env.mem None e (Name_var x) = true ->
        among x (Meet_env.names_to_avoid menv) = false
        ∧ among x (Meet_env.names_to_avoid joined) = true).
    { intros e Hin He. destruct (Hnew e x Hin He Hunknown) as [Hfresh Hamong].
      split; [ exact Hfresh | ].
      exact (meet_env_extends_avoid _ _ Jextends x Hamong). }
    destruct (Dmem x Hknown) as [ Hjoined_x | [Hin_vars Hkind_x] ].
    + destruct (known_in_some scoped_envs x) as [ (e & Hin & He) | Hnone ];
        [ exact (Hsome e Hin He) | ].
      destruct (Jnew x Hjoined_x Hunknown Hnone) as [Hnot Hyes].
      split; [ | exact Hyes ].
      destruct (among x (Meet_env.names_to_avoid menv)) eqn:Hmenv;
        [ | reflexivity ].
      change (among x (avoid state) = false) in Hnot.
      rewrite (Havoid x Hmenv) in Hnot. discriminate Hnot.
    + destruct (kind_of x) as [ k | ] eqn:Hk;
        [ | destruct Hkind_x; reflexivity ].
      destruct (Hvars x k Hk) as (e & Hin & Hlevel).
      destruct (Hscoped e Hin) as [Hextends_e Hfits_e].
      exact (Hsome e Hin
               (scoped_in_order_known tenv e Hfits_e Hextends_e x k Hlevel)).
  - (* What the environments kept know. *)
    intros stenv Hin Hcovered.
    destruct (Hscoped stenv Hin) as [Hextends_s Hfits_s].
    apply (covered_knows_after stenv kind_of (map fst extra) joined env_final
             Hfits_s Hextends_s Hknows0).
    + intros s kind Hknown Hs. exact (Jkinds stenv s kind Hin Hknown Hs).
    + intros y k Hk. assert (Hy := Hcovered y k Hk). split; [ | exact Hy ].
      apply in_map_iff. exists (y, k). split; [ reflexivity | ].
      exact (lookup_in y k extra Hy).
    + exact Hdefined.
  - (* No value is lost. *)
    intros stenv statics valuation valuation_k Hin Hstatics Hsatisfied
      Hextended_k Hsatisfied_k Hvalued_k.
    destruct (Hscoped stenv Hin) as [Hextends_s Hfits_s].
    destruct (Jsound stenv statics valuation_k Hin Hstatics
                (satisfied_strictly_extended statics tenv valuation valuation_k
                   (proj1 (proj1 Hfits)) Hextended_k Hsatisfied)
                Hsatisfied_k
                (Hvalued_k _ (increment_scope_valued tenv valuation
                                (proj1 Hfits))))
      as (valuation_j & Hextended_j & Hsatisfied_j & Hvalued_j).
    set (keep := fun x : variable =>
      Typing_env.mem None (Meet_env.typing_env joined) (Name_var x)
      || Typing_env.mem None stenv (Name_var x)).
    assert (Hrestricted :
      extended_from (Meet_env.typing_env joined) valuation_j
        (restrict_valuation keep valuation_j)).
    { intros x Hx. rewrite lookup_restrict_valuation. unfold keep.
      rewrite Hx. reflexivity. }
    assert (Hextended' :
      extended_from stenv valuation_k (restrict_valuation keep valuation_j)).
    { intros x Hx. rewrite lookup_restrict_valuation. unfold keep.
      rewrite Hx, orb_true_r. exact (Hextended_j x Hx). }
    exists (restrict_valuation keep valuation_j).
    split; [ exact Hextended' | ]. split.
    + apply (Dsatisfied statics).
      * exact (satisfied_strictly_extended statics _ valuation_j _
                 (proj1 (proj1 Jfits)) Hrestricted Hsatisfied_j).
      * intros y k v _ Hk Hunknown Hv.
        rewrite lookup_restrict_valuation in Hv. unfold keep in Hv.
        rewrite Hunknown in Hv. cbn [orb] in Hv.
        destruct (Typing_env.mem None stenv (Name_var y)) eqn:Hy;
          [ | discriminate Hv ].
        rewrite (Hextended_j y Hy) in Hv.
        destruct (Hvars y k Hk) as (e & Hin_e & Hlevel).
        destruct (Hscoped e Hin_e) as [Hextends_e Hfits_e].
        destruct (envs_apart_in _ Hapart e stenv Hin_e Hin)
          as [ -> | Hdisjoint ].
        -- exact (scoped_defined_value_kind tenv stenv Hfits_s Hextends_s
                    statics valuation_k y k v Hsatisfied_k Hlevel Hv).
        -- exfalso.
           rewrite (proj1 Hknows0 _
                      (Hdisjoint y
                         (scoped_in_order_known tenv e Hfits_e Hextends_e y k
                            Hlevel)
                         Hy))
             in Hunknown.
           discriminate Hunknown.
    + intros cut_after Hvalued. apply (Dvalued _).
      apply (equations_valued_extended (Meet_env.typing_env joined) cut_after
               valuation_j _ (proj1 Jfits) Hrestricted).
      apply Hvalued_j.
      exact (equations_valued_extended tenv cut_after valuation valuation_k
               (proj1 Hfits) Hextended_k Hvalued).
Qed.

(* ======================================================================== *)
(* The map of the cases                                                     *)
(* ======================================================================== *)

Variable is_empty_map_known : known -> bool.
Variable get_singleton_map_known : known -> option (row_tag * case).
Variable merge_map_known :
  (row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
   row_like_state -> option (option (Or_unknown.t case) * row_like_state)) ->
  known -> known -> row_like_state -> option (known * row_like_state).

(* The case that a map has for a tag, and what is asked of a map for
   it to be well formed. *)
Variable find : row_tag -> known -> option (Or_unknown.t case).
Variable known_ok : known -> Prop.

Hypothesis Hempty :
  ∀ cases : known,
    is_empty_map_known cases = true ->
    ∀ tag : row_tag, find tag cases = None.

(* The merge of two maps goes through the tags that either has, once
   each, with the state; the map answered is well formed and has, for
   a tag, what was answered for it. It is said with an invariant of
   the state, an order on states that the steps follow, and a fact
   about each tag, what was answered for it and the state then. *)
Hypothesis Hmerge :
  ∀ (merge_case :
       row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
       row_like_state -> option (option (Or_unknown.t case) * row_like_state))
    (known1 known2 cases : known) (state0 staten : row_like_state)
    (Inv : row_like_state -> Prop)
    (le : row_like_state -> row_like_state -> Prop)
    (P : row_tag -> option (Or_unknown.t case) -> row_like_state -> Prop),
    (∀ s : row_like_state, le s s) ->
    (∀ s1 s2 s3 : row_like_state, le s1 s2 -> le s2 s3 -> le s1 s3) ->
    Inv state0 ->
    (∀ (tag : row_tag) (s s' : row_like_state)
       (c : option (Or_unknown.t case)),
       Inv s ->
       merge_case tag (find tag known1) (find tag known2) s = Some (c, s') ->
       Inv s' ∧ le s s' ∧ P tag c s') ->
    merge_map_known merge_case known1 known2 state0 = Some (cases, staten) ->
    known_ok cases
    ∧ Inv staten
    ∧ le state0 staten
    ∧ (∀ tag : row_tag,
         find tag known1 <> None ∨ find tag known2 <> None ->
         ∃ st : row_like_state, le st staten ∧ P tag (find tag cases) st)
    ∧ ∀ tag : row_tag,
        find tag known1 = None -> find tag known2 = None ->
        find tag cases = None.

(* A row fits an environment: its map is well formed, and its cases
   fit. *)
Definition row_fits (stenv : Typing_env.t) (x : known * Or_bottom.t case)
    : Prop :=
  known_ok (fst x)
  ∧ (∀ (tag : row_tag) (c : case),
       find tag (fst x) = Some (Or_unknown.Known c) ->
       case_fits stenv (Some tag) c)
  ∧ ∀ c : case, snd x = Or_bottom.Ok c -> case_fits stenv None c.

(* It is true of the value in hand, whose tag is [tag0]: what the tag
   selects is. *)
Variable tag0 : row_tag.

Definition row_true (statics : static_env) (valuation : env)
    (x : known * Or_bottom.t case) : Prop :=
  sel_true statics valuation (selected (find tag0 (fst x)) (snd x)).

Lemma row_fits_kept (stenv stenv' : Typing_env.t)
    (x : known * Or_bottom.t case) :
  knows_at_kinds stenv stenv' -> row_fits stenv x -> row_fits stenv' x.
Proof.
  intros Hknows (Hok & Hcases & Hother). split; [ exact Hok | ]. split.
  - intros tag c Hc. exact (case_fits_kept _ _ _ c Hknows (Hcases tag c Hc)).
  - intros c Hc. exact (case_fits_kept _ _ _ c Hknows (Hother c Hc)).
Qed.

Lemma row_true_stable (statics : static_env) (stenv : Typing_env.t)
    (valuation valuation' : env) (x : known * Or_bottom.t case) :
  row_fits stenv x ->
  extended_from stenv valuation valuation' ->
  row_true statics valuation x ->
  row_true statics valuation' x.
Proof.
  intros (_ & Hcases & Hother) Hextended. unfold row_true.
  destruct (find tag0 (fst x)) as [ [ c | ] | ] eqn:Hfind; cbn.
  - exact (case_true_stable statics stenv valuation valuation' _ c
             (Hcases tag0 c Hfind) Hextended).
  - auto.
  - destruct (snd x) as [ c | ] eqn:Hsnd; cbn; [ | auto ].
    exact (case_true_stable statics stenv valuation valuation' _ c
             (Hother c eq_refl) Hextended).
Qed.

Lemma pair_sound_selected (s1 s2 r : option (Or_unknown.t case))
    (other : Or_bottom.t case) (state : row_like_state) :
  pair_sound s1 s2 r state -> pair_sound s1 s2 (selected r other) state.
Proof.
  intros Hpair statics valuation Hstatics Hsatisfied H1 H2.
  destruct (Hpair statics valuation Hstatics Hsatisfied H1 H2)
    as (stenv & valuation_k & Hin & Hextended & Hsatisfied_k & Hvalued & Hr).
  exists stenv, valuation_k. split; [ exact Hin | ].
  split; [ exact Hextended | ]. split; [ exact Hsatisfied_k | ].
  split; [ exact Hvalued | ].
  intros valuation' Hextended'. specialize (Hr valuation' Hextended').
  destruct r as [ c | ]; [ exact Hr | destruct Hr ].
Qed.

Lemma row_initial_inv : row_inv (row_initial_state base).
Proof.
  split.
  - intros stenv [].
  - apply Apart_nil.
  - intros stenv x [].
  - intros x Hx. exact Hx.
  - apply one_kind_nil.
  - intros y k Hy. discriminate Hy.
Qed.

(* The meet of two rows, in the environment [menv], whatever is taken
   for [need_join]. *)
Theorem row_like_body_ok (known1 known2 : known)
    (other1 other2 : Or_bottom.t case)
    (result : Meet_result.t (known * Or_bottom.t case)) :
  row_fits tenv (known1, other1) ->
  row_fits tenv (known2, other2) ->
  (let* '(cases, state) :=
     merge_map_known
       (fun _tag case1 case2 =>
          row_meet_knowns common_scope need_join base other1 other2 case1
            case2)
       known1 known2 (row_initial_state base)
   in
   let* '(other, state) :=
     row_meet_others common_scope need_join base other1 other2 state
   in
   row_finish is_empty_map_known menv common_scope tenv cases other state)
  = Some result ->
  meet_call_ok row_fits row_true menv (known1, other1) (known2, other2) result.
Proof.
  intros Hx1 Hx2 Hbody.
  assert (Hx1' := Hx1). assert (Hx2' := Hx2).
  destruct Hx1' as (_ & Hk1 & Ho1), Hx2' as (_ & Hk2 & Ho2).
  cbn [fst snd] in Hk1, Ho1, Hk2, Ho2.
  destruct (merge_map_known
              (fun _tag case1 case2 =>
                 row_meet_knowns common_scope need_join base other1 other2
                   case1 case2)
              known1 known2 (row_initial_state base))
    as [ [cases state_m] | ] eqn:Hfold; [ | discriminate Hbody ].
  cbv beta iota in Hbody.
  destruct (row_meet_others common_scope need_join base other1 other2 state_m)
    as [ [other state_f] | ] eqn:Hothers; [ | discriminate Hbody ].
  cbv beta iota in Hbody.
  set (P :=
    fun (tag : row_tag) (c : option (Or_unknown.t case))
        (st : row_like_state) =>
      result_placed (Some tag) c st
      ∧ (find tag known1 <> None ∨ find tag known2 <> None ->
         pair_sound (selected (find tag known1) other1)
           (selected (find tag known2) other2) c st)).
  assert (Hstep :
    ∀ (tag : row_tag) (s s' : row_like_state)
      (c : option (Or_unknown.t case)),
      row_inv s ->
      row_meet_knowns common_scope need_join base other1 other2
        (find tag known1) (find tag known2) s
      = Some (c, s') ->
      row_inv s' ∧ state_le s s' ∧ P tag c s').
  { intros tag s s' c Hinv Hanswer.
    assert (Hf1 : sel_fits tenv (Some tag) (find tag known1)).
    { destruct (find tag known1) as [ [ c1 | ] | ] eqn:Hfind; cbn; auto. }
    assert (Hf2 : sel_fits tenv (Some tag) (find tag known2)).
    { destruct (find tag known2) as [ [ c2 | ] | ] eqn:Hfind; cbn; auto. }
    destruct (row_meet_knowns_step tag other1 other2 _ _ s Hinv Hf1 Hf2 Ho1
                Ho2 c s' Hanswer)
      as (Hinv' & Hle & Hplaced & Hpair).
    split; [ exact Hinv' | ]. split; [ exact Hle | ].
    split; [ exact Hplaced | exact Hpair ]. }
  destruct (Hmerge _ known1 known2 cases _ state_m row_inv state_le P
              state_le_refl state_le_trans row_initial_inv Hstep Hfold)
    as (Hcases_ok & Hinv_m & _ & Hprocessed & Habsent).
  destruct (row_meet_others_step other1 other2 other state_m state_f Hinv_m
              Ho1 Ho2 Hothers)
    as (Hinv_f & Hle_f & Hother_placed & Hother_pair).
  (* The pair of the tag of the value. *)
  assert (Hvalue :
    pair_sound (selected (find tag0 known1) other1)
      (selected (find tag0 known2) other2) (selected (find tag0 cases) other)
      state_f).
  { destruct (find tag0 known1) as [ c1 | ] eqn:Hfind1.
    - destruct (Hprocessed tag0) as (st & Hle & _ & Hpair).
      { left. rewrite Hfind1. discriminate. }
      rewrite Hfind1 in Hpair. apply pair_sound_selected.
      apply (pair_sound_le _ _ _ st state_f
               (state_le_trans _ _ _ Hle Hle_f)).
      apply Hpair. left. discriminate.
    - destruct (find tag0 known2) as [ c2 | ] eqn:Hfind2.
      + destruct (Hprocessed tag0) as (st & Hle & _ & Hpair).
        { right. rewrite Hfind2. discriminate. }
        rewrite Hfind1, Hfind2 in Hpair. apply pair_sound_selected.
        apply (pair_sound_le _ _ _ st state_f
                 (state_le_trans _ _ _ Hle Hle_f)).
        apply Hpair. right. discriminate.
      + rewrite (Habsent tag0 Hfind1 Hfind2). exact Hother_pair. }
  (* Every case answered is placed. *)
  assert (Hplaced :
    ∀ (tag : row_tag) (c : case),
      find tag cases = Some (Or_unknown.Known c) ->
      case_placed (Some tag) c state_f).
  { intros tag c Hc.
    assert (Hbound : find tag known1 <> None ∨ find tag known2 <> None).
    { destruct (find tag known1) eqn:Hfind1; [ left; discriminate | ].
      destruct (find tag known2) eqn:Hfind2; [ right; discriminate | ].
      rewrite (Habsent tag Hfind1 Hfind2) in Hc. discriminate Hc. }
    destruct (Hprocessed tag Hbound) as (st & Hle & Hplaced_st & _).
    rewrite Hc in Hplaced_st.
    exact (result_placed_le (Some tag) (Some (Or_unknown.Known c)) st state_f
             (state_le_trans _ _ _ Hle Hle_f) Hplaced_st). }
  unfold row_finish in Hbody.
  destruct (is_empty_map_known cases
            && match other with
               | Or_bottom.Bottom => true
               | Or_bottom.Ok _ => false
               end)
    eqn:Hnothing.
  { injection Hbody as <-. apply meet_call_ok_bottom.
    intros statics valuation Hstatics Hsatisfied H1 H2.
    destruct (Hvalue statics valuation Hstatics Hsatisfied H1 H2)
      as (stenv & valuation_k & _ & _ & _ & _ & Hr).
    specialize (Hr valuation_k (extended_from_refl _ _)).
    apply andb_true_iff in Hnothing as [Hno_case Hno_other].
    rewrite (Hempty cases Hno_case tag0) in Hr.
    destruct other; [ discriminate Hno_other | exact Hr ]. }
  destruct (result_env state_f) as [ scoped_envs | ] eqn:Henvs.
  2:{ cbv beta iota in Hbody. injection Hbody as <-.
      apply meet_call_ok_bottom.
      intros statics valuation Hstatics Hsatisfied H1 H2.
      destruct (Hvalue statics valuation Hstatics Hsatisfied H1 H2)
        as (stenv & valuation_k & Hin & _).
      unfold envs_of in Hin. rewrite Henvs in Hin. destruct Hin. }
  assert (Henvs_of : envs_of state_f = scoped_envs).
  { unfold envs_of. rewrite Henvs. reflexivity. }
  destruct (row_join_all menv common_scope tenv scoped_envs state_f)
    as [ env_final | ] eqn:Hall; [ | discriminate Hbody ].
  cbv beta iota in Hbody. injection Hbody as <-.
  destruct (row_join_all_spec state_f scoped_envs env_final Hinv_f Henvs_of
              Hall)
    as (Hextends & Hfits_final & Hnames & Hknows & Hlost).
  assert (Hknows_initial :
    knows_at_kinds tenv (Meet_env.typing_env env_final)).
  { exact (meet_env_extends_knows _ _ Hextends). }
  (* A case that is placed fits the environment answered. *)
  assert (Hplaced_fits :
    ∀ (o : option row_tag) (c : case),
      case_placed o c state_f ->
      case_fits (Meet_env.typing_env env_final) o c).
  { intros o c [ Hc | (stenv & Hin & Hcovered & Hc) ].
    - exact (case_fits_kept _ _ o c Hknows_initial Hc).
    - rewrite Henvs_of in Hin.
      exact (case_fits_kept _ _ o c (Hknows stenv Hin Hcovered) Hc). }
  split.
  - intros r menv' [= <- <-].
    split; [ exact Hextends | ]. split; [ exact Hfits_final | ].
    split; [ | exact Hnames ].
    unfold row_match_with_input.
    destruct (result_is_t1 state_f), (result_is_t2 state_f);
      cbn [extract_value].
    + exact (row_fits_kept _ _ _ Hknows_initial Hx1).
    + exact (row_fits_kept _ _ _ Hknows_initial Hx1).
    + exact (row_fits_kept _ _ _ Hknows_initial Hx2).
    + split; [ exact Hcases_ok | ]. split.
      * intros tag c Hc. exact (Hplaced_fits _ c (Hplaced tag c Hc)).
      * intros c Hc. exact (Hplaced_fits _ c (Hother_placed c Hc)).
  - intros statics valuation Hstatics Hsatisfied H1 H2.
    destruct (Hvalue statics valuation Hstatics Hsatisfied H1 H2)
      as (stenv & valuation_k & Hin & Hextended_k & Hsatisfied_k & Hvalued_k
          & Hr).
    rewrite Henvs_of in Hin.
    destruct (Hlost stenv statics valuation valuation_k Hin Hstatics
                Hsatisfied Hextended_k Hsatisfied_k Hvalued_k)
      as (valuation' & Hextended' & Hsatisfied' & Hvalued').
    assert (Hextended :
      extended_from tenv valuation valuation').
    { intros x Hx.
      destruct (inv_scoped state_f Hinv_f stenv
                  ltac:(rewrite Henvs_of; exact Hin))
        as [Hscoped_s _].
      rewrite (Hextended' x (proj1 (scoped_knows tenv stenv Hscoped_s) _ Hx)).
      exact (Hextended_k x Hx). }
    eexists _, env_final, valuation'. split; [ reflexivity | ].
    split; [ exact Hextended | ]. split; [ exact Hsatisfied' | ].
    split; [ | exact Hvalued' ].
    unfold row_match_with_input.
    destruct (result_is_t1 state_f), (result_is_t2 state_f);
      cbn [extract_value].
    + exact (row_true_stable statics tenv valuation valuation' _ Hx1 Hextended
               H1).
    + exact (row_true_stable statics tenv valuation valuation' _ Hx1 Hextended
               H1).
    + exact (row_true_stable statics tenv valuation valuation' _ Hx2 Hextended
               H2).
    + exact (Hr valuation' Hextended').
Qed.

End Row.

(* ======================================================================== *)
(* The rows of blocks                                                       *)
(* ======================================================================== *)

(* What [meet_row_like_for_blocks] gives [meet_row_like]. *)
Definition block_meet_shape (shape1 shape2 : block_shape)
    : Or_bottom.t block_shape :=
  if block_shape_eqb shape1 shape2
  then Or_bottom.Ok shape1
  else Or_bottom.Bottom.

Definition block_merge_map_known
    (merge_case :
      Z -> option (Or_unknown.t row_like_block_case) ->
      option (Or_unknown.t row_like_block_case) -> row_like_state ->
      option (option (Or_unknown.t row_like_block_case) * row_like_state))
    (known1 known2 : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (state : row_like_state) :=
  merge_fold_failing Z.eqb
    (sort_numbers (Keyed.keys_of_both Z.eqb known1 known2))
    merge_case known1 known2 state.

(* What is taken for the parts of the general statement: the index of
   a case and its product are well formed together (conditions 6 and 7
   of the well-formedness of types), whatever the tag; the case of a
   tag is the one that the map binds it to; a map is well formed if it
   binds a tag once. *)
Definition block_index_ok (_ : option Z) (i : row_like_index nat block_shape)
    (m : list flambda_type) : Prop :=
  block_product_fits i m = true
  ∧ block_shape_canonical_b (index_shape i) = true.

Definition block_find (tag : Z)
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    : option (Or_unknown.t row_like_block_case) :=
  Keyed.lookup Z.eqb tag known.

Definition block_known_ok
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case)) : Prop :=
  Keyed.distinct Z.eqb (keys known) = true.

(* The types of the fields are true of the fields that there are. *)
Definition block_fields_true (fields : list value) (statics : static_env)
    (valuation : env) (tys : list flambda_type) : Prop :=
  fields_satisfied (type_satisfied statics valuation) tys fields.

(* A row of blocks fits, and is true of a block. *)
Definition block_row_fits
    : Typing_env.t ->
      Keyed.finite_map Z (Or_unknown.t row_like_block_case)
      * Or_bottom.t row_like_block_case -> Prop :=
  row_fits block_index_ok product_fits block_find block_known_ok.

Definition block_row_true (tag : Z) (fields : list value)
    : static_env -> env ->
      Keyed.finite_map Z (Or_unknown.t row_like_block_case)
      * Or_bottom.t row_like_block_case -> Prop :=
  row_true (block_index_satisfied tag fields) (block_fields_true fields)
    block_find tag.

(* The products. *)
Lemma product_part_stable (fields : list value) :
  part_stable product_fits (product_satisfied fields).
Proof.
  split.
  - intros menv menv' tys Hextends Hfits ty Hin.
    exact (type_fits_extends menv menv' ty Hextends (Hfits ty Hin)).
  - intros statics tenv valuation valuation' tys Hfits Hextended Hs i ty Hnth.
    destruct (Hs i ty Hnth) as (field & Hfield & Hty).
    exists field. split; [ exact Hfield | ].
    exact (type_satisfied_extended statics tenv valuation valuation' field ty
             (Hfits ty (nth_error_In _ _ Hnth)) Hextended Hty).
Qed.

Lemma product_fits_by_kinds : fits_by_kinds product_fits.
Proof.
  intros tenv tenv' tys Hknows Hfits ty Hin.
  exact (type_fits_by_kinds tenv tenv' ty Hknows (Hfits ty Hin)).
Qed.

Lemma block_fields_true_stable (fields : list value) (statics : static_env)
    (tenv : Typing_env.t) (valuation valuation' : env)
    (tys : list flambda_type) :
  product_fits tenv tys ->
  extended_from tenv valuation valuation' ->
  block_fields_true fields statics valuation tys ->
  block_fields_true fields statics valuation' tys.
Proof.
  intros Hfits Hextended. unfold block_fields_true. revert fields.
  induction tys as [ | ty tys IH ]; intros fields Hs; [ exact I | ].
  destruct fields as [ | field fields ]; [ exact I | ].
  destruct Hs as [Hty Hrest]. split.
  - exact (type_satisfied_extended statics tenv valuation valuation' field ty
             (Hfits ty (or_introl eq_refl)) Hextended Hty).
  - apply IH; [ | exact Hrest ]. intros ty' Hin. apply Hfits. right.
    exact Hin.
Qed.

(* The index of a case says that the block has a field for each type
   of the product. *)
Lemma block_fields_hold (tag : Z) (fields : list value) (statics : static_env)
    (valuation : env) (o : option Z) (i : row_like_index nat block_shape)
    (tys : list flambda_type) :
  block_index_ok o i tys ->
  block_index_satisfied tag fields i ->
  block_fields_true fields statics valuation tys ->
  product_satisfied fields statics valuation tys.
Proof.
  intros [Hfits _] [Hsize _] Hs. apply product_satisfied_fields_for.
  split; [ | exact Hs ]. unfold block_product_fits in Hfits.
  destruct (index_domain i) as [ size | size ].
  - apply Nat.eqb_eq in Hfits. lia.
  - apply Nat.leb_le in Hfits. lia.
Qed.

(* The index answered and the product answered are well formed
   together. *)
Lemma block_index_ok_meet (mh : meet_expanded_head) (o : option Z)
    (menv0 menv1 menv2 : Meet_env.t) (i1 i2 : row_like_index nat block_shape)
    (m1 m2 : list flambda_type)
    (ri : meet_return_value (row_like_index nat block_shape))
    (rm : meet_return_value (list flambda_type)) :
  block_index_ok o i1 m1 ->
  block_index_ok o i2 m2 ->
  row_meet_index Nat.eqb Nat.leb Nat.max block_meet_shape menv0 i1 i2
  = Meet_result.Ok ri menv1 ->
  meet_int_indexed_product mh menv1 m1 m2 = Some (Meet_result.Ok rm menv2) ->
  block_index_ok o (extract_value ri i1 i2) (extract_value rm m1 m2).
Proof.
  intros [Hf1 Hc1] [Hf2 Hc2] Hindex Hmeet.
  assert (Hlength :=
    meet_int_indexed_product_length_for mh menv1 menv2 m1 m2 _ rm Hmeet
      (extract_value_answer_for _ rm m1 m2)).
  unfold block_index_ok, block_product_fits in *. rewrite Hlength.
  unfold row_meet_index, block_meet_shape in Hindex.
  destruct (block_shape_eqb (index_shape i1) (index_shape i2)) eqn:Hshape;
    [ | discriminate Hindex ].
  destruct i1 as [d1 s1], i2 as [d2 s2]. cbn [index_domain index_shape] in *.
  destruct d1 as [ a | a ], d2 as [ b | b ].
  all: repeat match type of Hindex with
       | context [if ?test then _ else _] =>
         destruct test eqn:?; try discriminate Hindex
       end.
  all: injection Hindex as <- <-.
  all: cbn [extract_value index_domain index_shape].
  all: split; [ | assumption ].
  all: rewrite ?Nat.eqb_eq, ?Nat.leb_le, ?Nat.leb_gt in *; lia.
Qed.

(* Two indexes that a block both has meet, to an index that it has. *)
Lemma block_index_true_meet (tag : Z) (fields : list value) (o : option Z)
    (menv0 : Meet_env.t) (i1 i2 : row_like_index nat block_shape)
    (m1 m2 : list flambda_type) :
  block_index_ok o i1 m1 ->
  block_index_ok o i2 m2 ->
  block_index_satisfied tag fields i1 ->
  block_index_satisfied tag fields i2 ->
  ∃ ri : meet_return_value (row_like_index nat block_shape),
    row_meet_index Nat.eqb Nat.leb Nat.max block_meet_shape menv0 i1 i2
    = Meet_result.Ok ri menv0
    ∧ block_index_satisfied tag fields (extract_value ri i1 i2).
Proof.
  intros [_ Hc1] [_ Hc2] [Hd1 Hs1] [Hd2 Hs2].
  assert (Hsame : index_shape i1 = index_shape i2).
  { apply (block_shapes_disjoint _ _ tag fields); try assumption;
      apply block_shape_canonical_b_iff; assumption. }
  unfold row_meet_index, block_meet_shape. rewrite <- Hsame.
  rewrite (keyed_eqb_refl block_shape_eqb block_shape_eqb_eq).
  unfold block_index_satisfied.
  destruct i1 as [d1 s1], i2 as [d2 s2]. cbn [index_domain index_shape] in *.
  subst s2.
  destruct d1 as [ a | a ], d2 as [ b | b ].
  all: repeat match goal with
       | |- context [if ?test then _ else _] => destruct test eqn:?
       end.
  all: rewrite ?Nat.eqb_eq, ?Nat.leb_le, ?Nat.eqb_neq, ?Nat.leb_gt in *;
    try lia.
  all: eexists; (split; [ reflexivity | ]);
    cbn [extract_value index_domain index_shape];
    (split; [ lia | assumption ]).
Qed.

(* The merge of the maps of the cases, in the order of the tags. *)
Lemma block_merge_map_known_spec
    (merge_case :
      Z -> option (Or_unknown.t row_like_block_case) ->
      option (Or_unknown.t row_like_block_case) -> row_like_state ->
      option (option (Or_unknown.t row_like_block_case) * row_like_state))
    (known1 known2 cases :
      Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (state0 staten : row_like_state) (Inv : row_like_state -> Prop)
    (le : row_like_state -> row_like_state -> Prop)
    (P : Z -> option (Or_unknown.t row_like_block_case) -> row_like_state ->
         Prop) :
  (∀ s : row_like_state, le s s) ->
  (∀ s1 s2 s3 : row_like_state, le s1 s2 -> le s2 s3 -> le s1 s3) ->
  Inv state0 ->
  (∀ (tag : Z) (s s' : row_like_state)
     (c : option (Or_unknown.t row_like_block_case)),
     Inv s ->
     merge_case tag (block_find tag known1) (block_find tag known2) s
     = Some (c, s') ->
     Inv s' ∧ le s s' ∧ P tag c s') ->
  block_merge_map_known merge_case known1 known2 state0
  = Some (cases, staten) ->
  block_known_ok cases
  ∧ Inv staten
  ∧ le state0 staten
  ∧ (∀ tag : Z,
       block_find tag known1 <> None ∨ block_find tag known2 <> None ->
       ∃ st : row_like_state, le st staten ∧ P tag (block_find tag cases) st)
  ∧ ∀ tag : Z,
      block_find tag known1 = None -> block_find tag known2 = None ->
      block_find tag cases = None.
Proof.
  intros Hrefl Htrans Hinv Hstep Hmerge. unfold block_merge_map_known in Hmerge.
  set (keys := sort_numbers (Keyed.keys_of_both Z.eqb known1 known2)) in *.
  assert (Hperm : Permutation keys (Keyed.keys_of_both Z.eqb known1 known2))
    by apply sort_numbers_permutation.
  assert (Hnodup : NoDup keys).
  { apply (Permutation_NoDup (Permutation_sym Hperm)).
    apply (keyed_distinct_NoDup Z.eqb z_eqb_decides_equality).
    apply (keyed_keys_of_both_distinct Z Z.eqb z_eqb_decides_equality). }
  assert (Hin :
    ∀ tag : Z,
      tag ∈ keys
      <-> Keyed.is_bound Z.eqb tag known1 || Keyed.is_bound Z.eqb tag known2
          = true).
  { intros tag.
    rewrite <- (keyed_among_keys_of_both Z Z.eqb z_eqb_decides_equality),
      (keyed_among_in Z Z.eqb z_eqb_decides_equality).
    split; intros H;
      [ exact (Permutation_in _ Hperm H)
      | exact (Permutation_in _ (Permutation_sym Hperm) H) ]. }
  destruct (merge_fold_failing_spec Z.eqb z_eqb_decides_equality merge_case
              known1 known2 Inv le P Hrefl Htrans Hstep keys state0 staten
              cases Hnodup Hinv Hmerge)
    as (Hout & Hinv_n & Hle & Hkeys & Hrest).
  split.
  { exact (proj2 (keyed_distinct_NoDup Z.eqb z_eqb_decides_equality _) Hout). }
  split; [ exact Hinv_n | ]. split; [ exact Hle | ]. split.
  - intros tag Hbound. apply Hkeys, Hin.
    unfold Keyed.is_bound, block_find in *.
    destruct (Keyed.lookup Z.eqb tag known1); [ reflexivity | ].
    destruct (Keyed.lookup Z.eqb tag known2); [ reflexivity | ].
    destruct Hbound as [ H | H ]; destruct H; reflexivity.
  - intros tag H1 H2. apply Hrest. intros Hmem. apply Hin in Hmem.
    unfold Keyed.is_bound, block_find in *. rewrite H1, H2 in Hmem.
    discriminate Hmem.
Qed.

(* What it is for a row of blocks to fit, by its cases. *)
Lemma lists_fit_andb (tenv : Typing_env.t) (w w' : bool)
    (f : list variable) (a : list (simple * flambda_kind))
    (e : list (name * flambda_kind)) :
  lists_fit tenv (w && w') f a e <-> w = true ∧ lists_fit tenv w' f a e.
Proof. unfold lists_fit. rewrite andb_true_iff. tauto. Qed.

Lemma product_lists_fit (tenv : Typing_env.t) (tys : list flambda_type) :
  lists_fit tenv (every_b type_well_formed tys) (flat_map free_variables tys)
    (flat_map aliases_with_kinds tys)
    (flat_map equation_names_with_kinds tys)
  <-> product_fits tenv tys.
Proof.
  induction tys as [ | ty tys IH ].
  - split; [ intros _ ty [] | intros _; apply lists_fit_nil ].
  - cbn [forallb flat_map]. rewrite lists_fit_app, IH, <- type_fits_lists.
    unfold product_fits. split.
    + intros [Hty Htys] ty' [ <- | Hin ]; [ exact Hty | exact (Htys ty' Hin) ].
    + intros H. split; [ apply H; left; reflexivity | ].
      intros ty' Hin. apply H. right. exact Hin.
Qed.

Lemma block_case_lists (tenv : Typing_env.t) (o : option Z)
    (c : row_like_block_case) :
  lists_fit tenv
    (block_case_well_formed type_well_formed env_extension_well_formed c)
    (free_variables_of_block_case free_variables
       free_variables_of_env_extension c)
    (aliases_of_block_case aliases_with_kinds aliases_of_env_extension c)
    (equation_names_of_block_case equation_names_with_kinds
       equation_names_of_env_extension c)
  <-> block_index_ok o (case_index c) (case_maps_to c)
      ∧ product_fits tenv (case_maps_to c)
      ∧ lists_fit tenv (env_extension_well_formed (case_env_extension c))
          (free_variables_of_env_extension (case_env_extension c))
          (aliases_of_env_extension (case_env_extension c))
          (equation_names_of_env_extension (case_env_extension c)).
Proof.
  unfold block_case_well_formed, free_variables_of_block_case,
    aliases_of_block_case, equation_names_of_block_case, block_index_ok.
  rewrite lists_fit_app, <- andb_assoc, !lists_fit_andb, product_lists_fit.
  tauto.
Qed.

Lemma block_case_fits_of_lists (tenv : Typing_env.t) (o : option Z)
    (c : row_like_block_case) :
  typing_env_invariant tenv ->
  lists_fit tenv
    (block_case_well_formed type_well_formed env_extension_well_formed c)
    (free_variables_of_block_case free_variables
       free_variables_of_env_extension c)
    (aliases_of_block_case aliases_with_kinds aliases_of_env_extension c)
    (equation_names_of_block_case equation_names_with_kinds
       equation_names_of_env_extension c) ->
  case_fits block_index_ok product_fits tenv o c.
Proof.
  intros Hinvariant H.
  apply (block_case_lists tenv o c) in H as (Hindex & Hmaps & Hextension).
  split; [ exact Hindex | ]. split; [ exact Hmaps | ].
  exact (lists_fit_extension tenv _ Hinvariant Hextension).
Qed.

Lemma block_case_lists_of_fits (tenv : Typing_env.t) (o : option Z)
    (c : row_like_block_case) :
  case_fits block_index_ok product_fits tenv o c ->
  lists_fit tenv
    (block_case_well_formed type_well_formed env_extension_well_formed c)
    (free_variables_of_block_case free_variables
       free_variables_of_env_extension c)
    (aliases_of_block_case aliases_with_kinds aliases_of_env_extension c)
    (equation_names_of_block_case equation_names_with_kinds
       equation_names_of_env_extension c).
Proof.
  intros (Hindex & Hmaps & Hextension). apply (block_case_lists tenv o c).
  split; [ exact Hindex | ]. split; [ exact Hmaps | ].
  exact (extension_lists_fit tenv _ Hextension).
Qed.

(* The lists of the cases of a map that binds a tag once are those of
   each case that it binds a tag to. *)
Lemma block_known_lists (tenv : Typing_env.t)
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case)) :
  block_known_ok known ->
  lists_fit tenv
    (Keyed.for_all_bindings_b Z.eqb
       (fun _ case =>
          match case with
          | Or_unknown.Known case =>
            block_case_well_formed type_well_formed env_extension_well_formed
              case
          | Or_unknown.Unknown => true
          end)
       known)
    (flat_map
       (fun case =>
          match snd case with
          | Or_unknown.Known case =>
            free_variables_of_block_case free_variables
              free_variables_of_env_extension case
          | Or_unknown.Unknown => []
          end)
       known)
    (flat_map
       (fun case =>
          match snd case with
          | Or_unknown.Known case =>
            aliases_of_block_case aliases_with_kinds aliases_of_env_extension
              case
          | Or_unknown.Unknown => []
          end)
       known)
    (flat_map
       (fun case =>
          match snd case with
          | Or_unknown.Known case =>
            equation_names_of_block_case equation_names_with_kinds
              equation_names_of_env_extension case
          | Or_unknown.Unknown => []
          end)
       known)
  <-> ∀ (tag : Z) (c : row_like_block_case),
        block_find tag known = Some (Or_unknown.Known c) ->
        lists_fit tenv
          (block_case_well_formed type_well_formed env_extension_well_formed
             c)
          (free_variables_of_block_case free_variables
             free_variables_of_env_extension c)
          (aliases_of_block_case aliases_with_kinds aliases_of_env_extension
             c)
          (equation_names_of_block_case equation_names_with_kinds
             equation_names_of_env_extension c).
Proof.
  intros Hdistinct.
  assert (Hin :=
    in_lookup_distinct Z.eqb z_eqb_decides_equality known Hdistinct).
  assert (Hlookup :=
    fun tag c => keyed_lookup_in Z.eqb z_eqb_decides_equality tag c known).
  unfold lists_fit, block_find.
  rewrite (keyed_for_all_bindings_b Z Z.eqb z_eqb_decides_equality),
    (keyed_for_all_bindings_lookup Z Z.eqb z_eqb_decides_equality).
  split.
  - intros (Hwf & Hfv & Hal & Heq) tag c Hc.
    assert (Hmem := Hlookup tag _ Hc).
    split; [ exact (Hwf tag _ Hc) | ]. split; [ | split ].
    + intros x Hx. apply Hfv, in_flat_map. exists (tag, Or_unknown.Known c).
      split; [ exact Hmem | exact Hx ].
    + intros s k Hs. apply Hal, in_flat_map.
      exists (tag, Or_unknown.Known c). split; [ exact Hmem | exact Hs ].
    + intros n k Hn. apply Heq, in_flat_map.
      exists (tag, Or_unknown.Known c). split; [ exact Hmem | exact Hn ].
  - intros Hcases. split; [ | split; [ | split ] ].
    + intros tag [ c | ] Hc; [ exact (proj1 (Hcases tag c Hc)) | reflexivity ].
    + intros x Hx.
      apply in_flat_map in Hx as ([tag [ c | ]] & Hmem & Hx); [ | destruct Hx ].
      exact (proj1 (proj2 (Hcases tag c (Hin tag _ Hmem))) x Hx).
    + intros s k Hs.
      apply in_flat_map in Hs as ([tag [ c | ]] & Hmem & Hs); [ | destruct Hs ].
      exact (proj1 (proj2 (proj2 (Hcases tag c (Hin tag _ Hmem)))) s k Hs).
    + intros n k Hn.
      apply in_flat_map in Hn as ([tag [ c | ]] & Hmem & Hn); [ | destruct Hn ].
      exact (proj2 (proj2 (proj2 (Hcases tag c (Hin tag _ Hmem)))) n k Hn).
Qed.

(* A row of blocks fits as the type that it makes does. *)
Lemma blocks_type_fits_row (tenv : Typing_env.t)
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other : Or_bottom.t row_like_block_case) (alloc_mode : Alloc_mode.t) :
  typing_env_invariant tenv ->
  type_fits tenv (blocks_type (Row_like_for_blocks known other alloc_mode)) ->
  block_row_fits tenv (known, other).
Proof.
  intros Hinvariant Hfits. apply blocks_type_lists_fit in Hfits.
  cbn [row_like_for_blocks_well_formed free_variables_of_row_like_for_blocks
       aliases_of_row_like_for_blocks equation_names_of_row_like_for_blocks]
    in Hfits.
  apply lists_fit_app in Hfits as [Hknown Hother].
  apply lists_fit_andb in Hknown as [Hdistinct Hknown].
  split; [ exact Hdistinct | ]. split.
  - intros tag c Hc.
    apply (block_case_fits_of_lists tenv (Some tag) c Hinvariant).
    exact (proj1 (block_known_lists tenv known Hdistinct) Hknown tag c Hc).
  - intros c Hc. cbn [snd] in Hc. subst other.
    exact (block_case_fits_of_lists tenv None c Hinvariant Hother).
Qed.

Lemma block_row_fits_type (tenv : Typing_env.t)
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other : Or_bottom.t row_like_block_case) (alloc_mode : Alloc_mode.t) :
  block_row_fits tenv (known, other) ->
  type_fits tenv (blocks_type (Row_like_for_blocks known other alloc_mode)).
Proof.
  intros (Hok & Hknown & Hother). cbn [fst snd] in Hok, Hknown, Hother.
  apply blocks_type_lists_fit.
  cbn [row_like_for_blocks_well_formed free_variables_of_row_like_for_blocks
       aliases_of_row_like_for_blocks equation_names_of_row_like_for_blocks].
  apply lists_fit_app. split.
  - apply lists_fit_andb. split; [ exact Hok | ].
    apply (block_known_lists tenv known Hok). intros tag c Hc.
    exact (block_case_lists_of_fits tenv (Some tag) c (Hknown tag c Hc)).
  - destruct other as [ c | ]; [ | apply lists_fit_nil ].
    exact (block_case_lists_of_fits tenv None c (Hother c eq_refl)).
Qed.

(* A row of blocks is true of a block as the meaning of types says. *)
Lemma block_row_true_satisfied (statics : static_env) (valuation : env)
    (tag : Z) (fields : list value)
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other : Or_bottom.t row_like_block_case) (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_satisfied statics valuation tag fields
    (Row_like_for_blocks known other alloc_mode)
  <-> block_row_true tag fields statics valuation (known, other).
Proof.
  rewrite (row_like_for_blocks_find statics valuation tag fields known other
             alloc_mode).
  unfold block_row_true, row_true, block_find. cbn [fst snd].
  destruct (Keyed.lookup Z.eqb tag known) as [ [ c | ] | ]; cbn;
    [ reflexivity | reflexivity | ].
  destruct other as [ c | ]; cbn; reflexivity.
Qed.

Section Blocks.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.

Hypothesis Hjoin : join_scoped_envs_sound join.
Hypothesis Hdefine : define_again choices = false.
Hypothesis Hlift : lift_variables_of_single_case choices = true.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

(* The block in hand. *)
Variable tag : Z.
Variable fields : list value.

Lemma block_row_part_stable :
  part_stable block_row_fits (block_row_true tag fields).
Proof.
  split.
  - intros menv menv' x Hextends Hfits.
    exact (row_fits_kept block_index_ok product_fits product_fits_by_kinds
             block_find block_known_ok _ _ x
             (meet_env_extends_knows menv menv' Hextends) Hfits).
  - intros statics tenv valuation valuation' x Hfits Hextended Hs.
    exact (row_true_stable block_index_ok product_fits
             (block_index_satisfied tag fields) (block_fields_true fields)
             (block_fields_true_stable fields) block_find block_known_ok tag
             statics tenv valuation valuation' x Hfits Hextended Hs).
Qed.

(* [meet_row_like], as [meet_row_like_for_blocks] calls it, is right
   for the block in hand. *)
Lemma meet_block_rows_ok
    (get_singleton_map_known :
      Keyed.finite_map Z (Or_unknown.t row_like_block_case) ->
      option (Z * row_like_block_case)) :
  part_meet_ok block_row_fits (block_row_true tag fields)
    (fun (env : Meet_env.t)
         (left right :
           Keyed.finite_map Z (Or_unknown.t row_like_block_case)
           * Or_bottom.t row_like_block_case) =>
       meet_row_like join choices mh (meet_int_indexed_product mh) Nat.eqb
         Nat.leb Nat.max block_meet_shape Keyed.is_empty
         get_singleton_map_known block_merge_map_known env (fst left)
         (fst right) (snd left) (snd right)).
Proof.
  intros menv [known1 other1] [known2 other2] result Henv H1 H2 Hmeet.
  cbn [fst snd] in Hmeet. rewrite meet_row_like_unfold in Hmeet.
  refine (row_like_body_ok join choices mh (meet_int_indexed_product mh)
            Nat.eqb Nat.leb Nat.max block_meet_shape Hjoin Hdefine Hlift
            Hkeeps Hsound block_index_ok product_fits
            (block_index_satisfied tag fields) (block_fields_true fields)
            (product_satisfied fields) (product_part_stable fields)
            product_fits_by_kinds (block_fields_true_stable fields)
            (meet_int_indexed_product_ok_for mh fields Hkeeps Hsound) _ _ _ _ _
            menv Henv _ Keyed.is_empty block_merge_map_known block_find
            block_known_ok _ block_merge_map_known_spec tag known1 known2
            other1 other2 result H1 H2 Hmeet).
  - intros statics valuation o i m _ Hindex Htrue Hm.
    exact (block_fields_hold tag fields statics valuation o i m Hindex Htrue
             Hm).
  - intros statics valuation m Hm.
    exact (proj2 (proj1 (product_satisfied_fields_for fields statics valuation
                    m)
                    Hm)).
  - intros tag' i m H. exact H.
  - intros o menv0 menv1 menv2 i1 i2 m1 m2 ri rm.
    apply block_index_ok_meet.
  - intros o menv0 i1 i2 m1 m2. apply block_index_true_meet.
  - intros cases Hempty.
    exact (proj1 (keyed_is_empty_lookup Z Z.eqb z_eqb_decides_equality _
                    cases)
             Hempty).
Qed.

(* [meet_row_like_for_blocks] is right for the block in hand. *)
Lemma meet_row_like_for_blocks_at_ok :
  part_meet_ok (fits_as_type blocks_type)
    (fun (statics : static_env) (valuation : env)
         (blocks : row_like_for_blocks) =>
       row_like_for_blocks_satisfied statics valuation tag fields blocks)
    (meet_row_like_for_blocks join choices mh).
Proof.
  intros menv [known1 other1 alloc_mode1] [known2 other2 alloc_mode2] result
    Henv H1 H2 Hmeet.
  unfold meet_row_like_for_blocks in Hmeet. cbv zeta in Hmeet.
  assert (Hinvariant := proj1 Henv).
  refine (combine_results2_ok_for _ _ _ block_row_fits
            (block_row_true tag fields)
            (fun (_ : Typing_env.t) (_ : Alloc_mode.t) => True)
            (fun (_ : static_env) (_ : env) (_ : Alloc_mode.t) => True)
            (fits_as_type blocks_type)
            (fun (statics : static_env) (valuation : env)
                 (blocks : row_like_for_blocks) =>
               row_like_for_blocks_satisfied statics valuation tag fields
                 blocks)
            _ _
            (fun known_and_other alloc_mode =>
               Row_like_for_blocks (fst known_and_other)
                 (snd known_and_other) alloc_mode)
            menv (known1, other1) (known2, other2) alloc_mode1 alloc_mode2
            result block_row_part_stable alloc_mode_part_stable
            (meet_block_rows_ok _) meet_alloc_mode_ok _ _ Henv
            (blocks_type_fits_row _ known1 other1 alloc_mode1 Hinvariant H1)
            (blocks_type_fits_row _ known2 other2 alloc_mode2 Hinvariant H2)
            I I Hmeet).
  - intros tenv [known other] alloc_mode Hfits _.
    exact (block_row_fits_type tenv known other alloc_mode Hfits).
  - intros statics valuation [known other] alloc_mode. cbn [fst snd].
    rewrite block_row_true_satisfied. tauto.
Qed.

End Blocks.

(* [meet_row_like_for_blocks] is right, for any value: the block in
   hand is the one that the value is, which the static environment
   says. *)
Lemma meet_row_like_for_blocks_ok_for :
  for_validity meet_row_like_for_blocks_ok_statement.
Proof.
  intros join choices mh v Hjoin Hdefine Hlift Hkeeps Hsound.
  set (choose :=
    fun (statics : static_env) (block : Z * list value) =>
      block_of statics v = Some block).
  set (satisfied :=
    fun (block : Z * list value) (statics : static_env) (valuation : env)
        (blocks : row_like_for_blocks) =>
      row_like_for_blocks_satisfied statics valuation (fst block) (snd block)
        blocks).
  assert (Hfunctional :
    ∀ (statics : static_env) (block block' : Z * list value),
      choose statics block -> choose statics block' -> block = block').
  { intros statics block block' H H'. unfold choose in H, H'. congruence. }
  assert (Hiff :
    ∀ (statics : static_env) (valuation : env) (blocks : row_like_for_blocks),
      satisfied_as_type blocks_type v statics valuation blocks
      <-> chosen_satisfied choose satisfied statics valuation blocks).
  { intros statics valuation blocks.
    unfold satisfied_as_type, chosen_satisfied, choose, satisfied.
    rewrite (blocks_type_satisfied statics valuation v). split.
    - intros (tag & fields & Hblock & Hrow). exists (tag, fields).
      split; [ exact Hblock | exact Hrow ].
    - intros ([tag fields] & Hblock & Hrow). exists tag, fields.
      split; [ exact Hblock | exact Hrow ]. }
  assert (Hchosen :=
    part_meet_ok_chosen (fits_as_type blocks_type) choose satisfied
      (meet_row_like_for_blocks join choices mh) (0%Z, []) Hfunctional
      (fun block =>
         meet_row_like_for_blocks_at_ok join choices mh Hjoin Hdefine Hlift
           Hkeeps Hsound (fst block) (snd block))).
  intros menv blocks1 blocks2 result Henv H1 H2 Hmeet.
  apply (meet_call_ok_change _ _ _ _ menv blocks1 blocks2 blocks1 blocks2
           result (Hchosen menv blocks1 blocks2 result Henv H1 H2 Hmeet)).
  - intros r menv' _ H. exact H.
  - intros statics valuation. apply Hiff.
  - intros statics valuation. apply Hiff.
  - intros r statics valuation. apply Hiff.
Qed.

End Validity.

Theorem meet_row_like_for_blocks_ok :
  meet_row_like_for_blocks_ok_statement.
Proof.
  first
    [ exact (@meet_row_like_for_blocks_ok_for standard_validity)
    | exact meet_row_like_for_blocks_ok_for ].
Qed.
