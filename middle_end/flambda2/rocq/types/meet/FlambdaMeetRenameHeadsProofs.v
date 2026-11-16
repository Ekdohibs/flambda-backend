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


(* For the proofs of the statements of FlambdaMeetRenameSpec.v: the
   meets of the heads of types under a renaming, from the rows of
   blocks and of closures to the meet of two expanded types. Nothing
   here needs review. The laws are for two meets and two joins that
   one renaming relates; the primed one is given the renamed
   arguments. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeRenameSpec FlambdaTypeRenameProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaTypingEnv
  FlambdaEnvRenameSpec FlambdaEnvRenameProofs FlambdaMeetGrammar
  FlambdaExpandHead FlambdaProversRenameSpec FlambdaProversRenameProofs
  FlambdaTypeAccessProofs FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetRenameSpec FlambdaEnvRenameMeetProofs
  FlambdaEnvRenameMeetPartsProofs FlambdaEnvRenameMeetProductsProofs
  FlambdaEnvRenameMeetValuesProofs FlambdaMeetRowLikeProofs
  FlambdaMeetRenameScopesProofs FlambdaMeetRenameRowLikeProofs.
Import ListNotations.

(* ======================================================================== *)
(* A merge of two maps that threads a state                                 *)
(* ======================================================================== *)

Lemma is_empty_map_values {K A B : Type} (f : A -> B)
    (bindings : Keyed.finite_map K A) :
  Keyed.is_empty (Keyed.map_values f bindings) = Keyed.is_empty bindings.
Proof. destruct bindings; reflexivity. Qed.

Lemma keys_of_both_map_values {K A A' B B' : Type} (eqb : K -> K -> bool)
    (f : A -> A') (g : B -> B') (m1 : Keyed.finite_map K A)
    (m2 : Keyed.finite_map K B) :
  Keyed.keys_of_both eqb (Keyed.map_values f m1) (Keyed.map_values g m2)
  = Keyed.keys_of_both eqb m1 m2.
Proof.
  unfold Keyed.keys_of_both, Keyed.map_values. rewrite !map_map.
  reflexivity.
Qed.

Section Merge_renamed.

Context {K A B C S : Type}.

Variable eqb : K -> K -> bool.
Variable ga : A -> A.
Variable gb : B -> B.
Variable gc : C -> C.
Variable h : S -> S.
Variable f f' : K -> option A -> option B -> S -> option (option C * S).
Variable m1 : Keyed.finite_map K A.
Variable m2 : Keyed.finite_map K B.

Hypothesis Hf :
  ∀ (key : K) (a : option A) (b : option B) (s : S),
    f' key (option_map ga a) (option_map gb b) (h s)
    = option_map (fun res => (option_map gc (fst res), h (snd res)))
        (f key a b s).

Lemma failing_fold_renamed (keys : list K) :
  ∀ (out : Keyed.finite_map K C) (state : option S),
    List.fold_left
      (failing_step eqb f' (Keyed.map_values ga m1) (Keyed.map_values gb m2))
      keys (Keyed.map_values gc out, option_map h state)
    = (Keyed.map_values gc
         (fst (List.fold_left (failing_step eqb f m1 m2) keys (out, state))),
       option_map h
         (snd (List.fold_left (failing_step eqb f m1 m2) keys (out, state)))).
Proof.
  induction keys as [ | key keys IH]; intros out state; [ reflexivity | ].
  cbn [List.fold_left].
  assert (Hstep :
            failing_step eqb f' (Keyed.map_values ga m1)
              (Keyed.map_values gb m2)
              (Keyed.map_values gc out, option_map h state) key
            = (Keyed.map_values gc
                 (fst (failing_step eqb f m1 m2 (out, state) key)),
               option_map h
                 (snd (failing_step eqb f m1 m2 (out, state) key)))).
  { unfold failing_step. cbn [fst snd].
    destruct state as [state | ]; cbn [option_map]; [ | reflexivity ].
    rewrite !keyed_lookup_map_values, Hf.
    destruct (f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2)
                state) as [[[c | ] state'] | ]; cbn [option_map fst snd];
      try reflexivity.
    unfold Keyed.map_values. rewrite map_app. reflexivity. }
  rewrite Hstep.
  destruct (failing_step eqb f m1 m2 (out, state) key) as [out' state'].
  apply IH.
Qed.

Lemma merge_fold_failing_renamed (keys : list K) (init : S) :
  merge_fold_failing eqb keys f' (Keyed.map_values ga m1)
    (Keyed.map_values gb m2) (h init)
  = option_map (fun res => (Keyed.map_values gc (fst res), h (snd res)))
      (merge_fold_failing eqb keys f m1 m2 init).
Proof.
  rewrite !merge_fold_failing_unfold.
  change (@nil (K * C), Some (h init))
    with (Keyed.map_values gc (@nil (K * C)), option_map h (Some init)).
  rewrite failing_fold_renamed.
  match goal with
  | |- context [snd ?x] =>
    match goal with
    | |- context [match ?y with pair _ _ => _ end] => change y with x
    end;
    destruct x as [out [state | ]]; reflexivity
  end.
Qed.

End Merge_renamed.

(* ======================================================================== *)
(* The rows of blocks and of closures                                       *)
(* ======================================================================== *)

Section Heads.

Variable r : renaming.
Variables join join' : join_scoped_envs.
Variable choices : meet_choices.
Variables m m' : meet_expanded_head.

Hypothesis Hjoin : join_related_at r join join' m m'.
Hypothesis Hm : meet_expanded_head_related r m m'.

Hypothesis Hchoices : meet_choices_closed choices.

Lemma meet_row_like_for_blocks_renamed :
  meet_of_parts_related r (rename_row_like_for_blocks r)
    (rename_row_like_for_blocks r) (meet_row_like_for_blocks join choices m)
    (meet_row_like_for_blocks join' choices m').
Proof.
  intros env [known1 other1 alloc_mode1] [known2 other2 alloc_mode2].
  cbn [rename_row_like_for_blocks]. unfold meet_row_like_for_blocks.
  set (rm := fun r : renaming => map (rename_type r)).
  set (rk :=
         fun (r : renaming)
             (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case)) =>
           Keyed.map_values (or_unknown_map (rename_case rm r)) known).
  set (fa :=
         fun x : Keyed.finite_map Z (Or_unknown.t row_like_block_case)
                 * Or_bottom.t row_like_block_case =>
           (rk r (fst x), or_bottom_map (rename_case rm r) (snd x))).
  change (Keyed.map_values
            (or_unknown_map
               (rename_block_case (rename_type r) (rename_env_extension r)))
            known1,
          or_bottom_map
            (rename_block_case (rename_type r) (rename_env_extension r))
            other1)
    with (fa (known1, other1)).
  change (Keyed.map_values
            (or_unknown_map
               (rename_block_case (rename_type r) (rename_env_extension r)))
            known2,
          or_bottom_map
            (rename_block_case (rename_type r) (rename_env_extension r))
            other2)
    with (fa (known2, other2)).
  apply (combine_results2_renamed r fa (fun alloc_mode => alloc_mode)
           (rename_row_like_for_blocks r)).
  - intros env' [k1 o1] [k2 o2]. unfold fa. cbn [fst snd].
    apply (meet_row_like_renamed r join join' choices m m' Hjoin Hm)
      with (rk := rk).
    + exact (meet_int_indexed_product_renamed r m m' Hm).
    + intros k. apply is_empty_map_values.
    + intros k. unfold rk.
      rewrite (keyed_get_singleton_binding_map_values Z.eqb Z.eqb_eq).
      unfold row_like_block_case.
      match goal with
      | |- context [@Keyed.get_singleton_binding ?K ?eqb ?A k] =>
        destruct (@Keyed.get_singleton_binding K eqb A k)
          as [[tag [c | ]] | ];
          reflexivity
      end.
    + intros f f' k1' k2' state Hf. unfold rk.
      rewrite keys_of_both_map_values.
      exact (merge_fold_failing_renamed Z.eqb _ _ _ _ f f' k1' k2' Hf _ state).
  - intros env' mode1 mode2.
    rewrite (meet_alloc_mode_renamed r (fun alloc_mode => alloc_mode)).
    reflexivity.
  - intros [k o] alloc_mode. reflexivity.
Qed.

Lemma meet_row_like_for_closures_renamed :
  meet_of_parts_related r (rename_row_like_for_closures r)
    (rename_row_like_for_closures r)
    (meet_row_like_for_closures join choices m)
    (meet_row_like_for_closures join' choices m').
Proof.
  intros env [known1] [known2].
  cbn [rename_row_like_for_closures]. unfold meet_row_like_for_closures.
  set (rk :=
         fun (r : renaming) (known : finite_map row_like_closures_case) =>
           Keyed.map_values (rename_case rename_closures_entry r) known).
  pose proof
    (meet_row_like_renamed r join join' choices m m' Hjoin Hm
       rename_closures_entry rk (meet_closures_entry m)
       (meet_closures_entry m') set_of_closures_contents_equal
       set_of_closures_contents_subset set_of_closures_contents_union
       (fun _ _ : unit => Or_bottom.Ok tt) Keyed.is_empty
       (Keyed.get_singleton_binding Pos.eqb))
    as Hrow.
  match goal with
  | |- match meet_row_like _ _ _ _ _ _ _ _ _ _ ?merge _ _ _ _ _ with
       | _ => _
       end = _ =>
    specialize (Hrow merge)
  end.
  specialize (Hrow (meet_closures_entry_renamed r m m' Hm)).
  specialize (Hrow (fun k => is_empty_map_values _ k)).
  specialize
    (Hrow
       (fun k =>
          keyed_get_singleton_binding_map_values Pos.eqb Pos.eqb_eq _ k)).
  assert (Hmerge :
            ∀ (f f' :
                function_slot ->
                option (Or_unknown.t row_like_closures_case) ->
                option (Or_unknown.t row_like_closures_case) ->
                row_like_state ->
                option
                  (option (Or_unknown.t row_like_closures_case)
                   * row_like_state))
              (k1 k2 : finite_map row_like_closures_case)
              (state : row_like_state),
              (∀ (tag : function_slot)
                 (c1 c2 : option (Or_unknown.t row_like_closures_case))
                 (state : row_like_state),
                f' tag
                  (option_map
                     (or_unknown_map (rename_case rename_closures_entry r))
                     c1)
                  (option_map
                     (or_unknown_map (rename_case rename_closures_entry r))
                     c2)
                  (rename_row_like_state r state)
                = option_map
                    (fun res =>
                       (option_map
                          (or_unknown_map
                             (rename_case rename_closures_entry r))
                          (fst res),
                        rename_row_like_state r (snd res)))
                    (f tag c1 c2 state)) ->
              merge_fold_failing Pos.eqb
                (Keyed.keys_of_both Pos.eqb (rk r k1) (rk r k2))
                (fun fslot case1 case2 state =>
                   let case1 := option_map Or_unknown.Known case1 in
                   let case2 := option_map Or_unknown.Known case2 in
                   let* '(case, state) := f' fslot case1 case2 state in
                   match case with
                   | None => Some (None, state)
                   | Some (Or_unknown.Known case) => Some (Some case, state)
                   | Some Or_unknown.Unknown => None
                   end)
                (rk r k1) (rk r k2) (rename_row_like_state r state)
              = option_map
                  (fun res =>
                     (rk r (fst res), rename_row_like_state r (snd res)))
                  (merge_fold_failing Pos.eqb
                     (Keyed.keys_of_both Pos.eqb k1 k2)
                     (fun fslot case1 case2 state =>
                        let case1 := option_map Or_unknown.Known case1 in
                        let case2 := option_map Or_unknown.Known case2 in
                        let* '(case, state) := f fslot case1 case2 state in
                        match case with
                        | None => Some (None, state)
                        | Some (Or_unknown.Known case) =>
                          Some (Some case, state)
                        | Some Or_unknown.Unknown => None
                        end)
                     k1 k2 state)).
  { intros f f' k1 k2 state Hf. unfold rk.
    rewrite keys_of_both_map_values.
    apply merge_fold_failing_renamed. intros key a b s. cbv zeta.
    pose proof
      (Hf key (option_map Or_unknown.Known a) (option_map Or_unknown.Known b)
         s)
      as Hf'.
    destruct a as [a | ], b as [b | ];
      cbn [option_map or_unknown_map] in Hf' |- *;
      match goal with
      | |- match ?lhs with _ => _ end = _ =>
        match type of Hf' with
        | _ = ?rhs => replace lhs with rhs by (symmetry; exact Hf')
        end
      end;
      match goal with
      | |- context [option_map _ ?x] =>
        lazymatch x with
        | f _ _ _ _ => destruct x as [[[[c | ] | ] s'] | ]; reflexivity
        end
      end. }
  specialize (Hrow Hmerge env known1 known2 Or_bottom.Bottom
                Or_bottom.Bottom).
  cbn [or_bottom_map] in Hrow. unfold rk in Hrow.
  match goal with
  | |- match ?lhs with _ => _ end = _ =>
    match type of Hrow with
    | _ = ?rhs => replace lhs with rhs by (symmetry; exact Hrow)
    end
  end.
  clear Hrow Hmerge.
  match goal with
  | |- match option_map _ ?x with _ => _ end
       = option_map _ (match ?y with _ => _ end) =>
    change y with x; destruct x as [[res | res env'] | ]; try reflexivity
  end.
  destruct res as [ | | | [known [other | ]]]; reflexivity.
Qed.

(* ======================================================================== *)
(* The meet of two expanded types                                           *)
(* ======================================================================== *)

(* The heads of kind value, from the two rows
   (FlambdaEnvRenameMeetValuesProofs.v). *)
Lemma value_heads_renamed :
  meet_of_parts_related r (rename_head_of_kind_value r)
    (rename_head_of_kind_value r) (meet_head_of_kind_value join choices m)
    (meet_head_of_kind_value join' choices m').
Proof.
  exact (meet_head_of_kind_value_renamed r join join' choices m m' Hjoin Hm
           meet_row_like_for_blocks_renamed
           meet_row_like_for_closures_renamed).
Qed.

Lemma meet_expanded_head0_renamed (env : Meet_env.t)
    (head1 head2 : Expanded_type.head) :
  meet_expanded_head0 join' choices m' (rename_meet_env r env)
    (rename_expanded_head r head1) (rename_expanded_head r head2)
  = option_map (rename_meet_result r (rename_expanded r))
      (meet_expanded_head0 join choices m env head1 head2).
Proof.
  destruct head1 as [head1 | head1 | k1 head1 | head1 | head1],
    head2 as [head2 | head2 | k2 head2 | head2 | head2];
    cbn [rename_expanded_head meet_expanded_head0]; try reflexivity.
  - rewrite value_heads_renamed.
    destruct (meet_head_of_kind_value join choices m env head1 head2)
      as [result | ]; cbn [option_map]; [ | reflexivity ].
    f_equal. apply map_result_renamed. intros head. reflexivity.
  - rewrite
      (meet_head_of_kind_naked_immediate_renamed r m m' Hm choices Hchoices).
    destruct (meet_head_of_kind_naked_immediate choices m env head1 head2)
      as [result | ]; cbn [option_map]; [ | reflexivity ].
    f_equal. apply map_result_renamed. intros head. reflexivity.
  - match goal with
    | |- context [if ?c then _ else _] =>
      match type of c with
      | bool => destruct c; [ | reflexivity ]
      end
    end.
    cbn [option_map]. f_equal.
    rewrite (meet_head_of_kind_naked_number_renamed r (fun s => s));
      [ | reflexivity ].
    apply map_result_renamed. intros head. reflexivity.
Qed.

Lemma meet_expanded_head_step_renamed :
  meet_expanded_head_related r (meet_expanded_head_step join choices m)
    (meet_expanded_head_step join' choices m').
Proof.
  intros env expanded1 expanded2. unfold meet_expanded_head_step.
  change (Expanded_type.descr (rename_expanded r expanded1))
    with (or_unknown_or_bottom_map (rename_expanded_head r)
            (Expanded_type.descr expanded1)).
  change (Expanded_type.descr (rename_expanded r expanded2))
    with (or_unknown_or_bottom_map (rename_expanded_head r)
            (Expanded_type.descr expanded2)).
  apply meet_or_unknown_or_bottom_renamed. intros env' head1 head2.
  apply meet_expanded_head0_renamed.
Qed.

End Heads.
