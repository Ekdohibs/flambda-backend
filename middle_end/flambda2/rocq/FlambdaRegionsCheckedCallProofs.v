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

(* Lemmas for the proofs of the statements of FlambdaRegionsCheckedSpec.v:
   a call from a caller whose state passed the check comes to [entered]: a
   call of a callee that is not known, of a closure or of a partial
   application, with as many arguments as the code takes, with fewer or
   with more; and a direct call. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs FlambdaValueSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsFreshProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedUnknownProofs
  FlambdaRegionsCheckedParamsProofs FlambdaRegionsCheckedJumpProofs
  FlambdaRegionsCheckedStateProofs.
Import ListNotations.

(* ======================================================================== *)
(* Arities                                                                  *)
(* ======================================================================== *)

Lemma unarize_concat (a : arity) :
  unarize a = concat (unarize_per_parameter a).
Proof. unfold unarize, unarize_per_parameter. apply flat_map_concat_map. Qed.

Lemma unarize_app (a b : arity) : unarize (a ++ b) = unarize a ++ unarize b.
Proof. unfold unarize. apply flat_map_app. Qed.

Lemma agree_prefix (params : list (list flambda_kind)) :
  ∀ args : list (list flambda_kind),
    kinds_agree_on_prefix params args = true ->
    length args <= length params ->
    args = firstn (length args) params.
Proof.
  induction params as [ | param params IH]; intros [ | arg args];
    cbn [kinds_agree_on_prefix length firstn]; try reflexivity; try lia.
  intros H Hlength. apply andb_true_iff in H as [Hhead Hrest].
  apply kinds_eqb_eq in Hhead. subst arg. f_equal.
  apply IH; [ exact Hrest | lia ].
Qed.

Lemma concat_firstn_length {A : Type} (l : list (list A)) :
  ∀ n : nat, length (concat (firstn n l)) <= length (concat l).
Proof.
  induction l as [ | a l IH]; intros [ | n]; cbn [firstn concat length];
    try lia.
  rewrite !length_app. specialize (IH n). lia.
Qed.

(* Arguments that agree with the parameters as far as they go, and are
   no more than they, are no more values than the parameters. *)
Lemma prefix_cardinal (params_arity args_arity : arity) :
  kinds_agree_on_prefix (unarize_per_parameter params_arity)
    (unarize_per_parameter args_arity)
  = true ->
  length args_arity <= length params_arity ->
  length (unarize args_arity) <= cardinal_unarized params_arity.
Proof.
  intros Hagree Hlength. unfold cardinal_unarized. rewrite !unarize_concat.
  pose proof (agree_prefix _ _ Hagree) as Hprefix.
  unfold unarize_per_parameter in Hprefix at 1 2.
  rewrite !length_map in Hprefix. rewrite (Hprefix Hlength).
  apply concat_firstn_length.
Qed.

(* ======================================================================== *)
(* The shape of a call                                                      *)
(* ======================================================================== *)

Lemma shape_of_partial_callee (statics : static_env) (closure : value)
    (applied : list value) (applied_arity : arity) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) (shape : call_shape) :
  shape_of_call statics
    (Value_partial_application closure applied applied_arity) kind vs
    args_arity return_arity
  = Some shape ->
  kind = Indirect
  ∧ args_arity <> []
  ∧ map kind_of_value vs = unarize args_arity
  ∧ shape_of_closure_call statics closure Indirect (applied ++ vs)
      (applied_arity ++ args_arity) return_arity
    = Some shape.
Proof.
  unfold shape_of_call. cbn [partial_application_of].
  destruct kind as [named | ]; [ discriminate | ].
  destruct args_arity as [ | first args_arity]; [ discriminate | ].
  destruct (kinds_eqb (map kind_of_value vs) (unarize (first :: args_arity)))
    eqn:Hkinds; [ | discriminate ].
  intros H. split; [ reflexivity | ]. split; [ discriminate | ].
  split; [ apply kinds_eqb_eq; exact Hkinds | exact H ].
Qed.

(* ======================================================================== *)
(* What a call reads                                                        *)
(* ======================================================================== *)

Section Reads.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

Hypothesis Hfresh : symbols_fresh rs = true.

Hypothesis Hlocal : local_closures_fit cert rs = true.

Hypothesis Hdeep : frames_deep rs (rs_frames rs) = true.

Hypothesis Hstatics : statics_deep rs = true.

Hypothesis Hgood_statics :
  Forall (fun binding => allocated_values (good cert codes rs) (snd binding))
    (static_values (live_statics rs)).

(* The fields of a tuple that is good are. *)
Lemma good_fields : P_fields (good cert codes rs) (live_statics rs).
Proof.
  intros size tuple fields Htuple Hfields.
  apply (tuple_fields_of_blocks (good cert codes rs) (live_statics rs) size
           tuple fields);
    [ | exact Hfields ].
  intros tag block Hblock.
  destruct (FlambdaSemanticsProofs.block_of_cases _ _ _ _ Hblock)
    as [-> | (s & -> & Hlookup)].
  - destruct Htuple as [Hplain Hfits].
    cbn [value_plain value_all value_paps] in *.
    apply Forall_forall. intros w Hin.
    split;
      [ apply all_plain; exact (forallb_in _ _ w Hplain Hin)
      | exact (forallb_in _ _ w Hfits Hin) ].
  - apply lookup_in in Hlookup. rewrite Forall_forall in Hgood_statics.
    exact (Hgood_statics (s, Static_block tag block) Hlookup).
Qed.

(* What the frames bind reaches symbols that may be evaluated. *)
Lemma frames_within :
  ∀ above rest : list frame,
    rs_frames rs = above ++ rest ->
    frames_deep rs rest = true ->
    forallb
      (fun '(_, allocated) => allocated_all (symbol_within rs None) allocated)
      (live_locals rest)
    = true.
Proof.
  intros above rest. revert above.
  induction rest as [ | f below IH]; intros above Hframes;
    [ reflexivity | ].
  cbn [frames_deep]. intros H. apply andb_true_iff in H as [Hf Hbelow].
  change (live_locals (f :: below)) with (frame_locals f ++ live_locals below).
  rewrite forallb_app. apply andb_true_iff. split.
  - apply forallb_forall. intros [s allocated] Hin.
    pose proof (forallb_in _ _ (s, allocated) Hf Hin) as Hall.
    cbv beta iota in Hall. revert Hall. apply allocated_all_mono.
    intros s0 H0. unfold symbol_within.
    apply orb_true_iff in H0 as [H0 | H0]; [ rewrite H0; reflexivity | ].
    rewrite Hframes, (local_in_below above _ s0 H0). apply orb_true_r.
  - apply (IH (above ++ [f])); [ | exact Hbelow ].
    rewrite Hframes, <- app_assoc. reflexivity.
Qed.

(* A closure that is called: it is live; and it reaches symbols of the
   heap only, unless a closure of its code may be local. *)
Lemma closure_place (closure : value) (id : code_id)
    (slots : list (value_slot * value)) (entry : code_entry) :
  closure_of (live_statics rs) closure = Some (id, slots) ->
  good cert codes rs closure ->
  lookup id (cert_code cert) = Some entry ->
  live rs closure
  ∧ (ce_closure_local entry = false ->
     value_all (heap_symbol rs) closure = true).
Proof.
  intros Hof [Hplain _] Hentry.
  destruct closure
    as [c | code slots0 | slot decls slots0 | s | tag fields | elements | c
       | l | callee args args_arity];
    cbn [closure_of] in Hof; try discriminate.
  - cbn [value_plain] in Hplain.
    split; [ exact (heap_within rs None _ Hplain) | intros _; exact Hplain ].
  - cbn [value_plain] in Hplain.
    split; [ exact (heap_within rs None _ Hplain) | intros _; exact Hplain ].
  - destruct (lookup s (static_values (live_statics rs))) as [allocated | ]
      eqn:Hs; [ | discriminate ].
    unfold live_statics in Hs. cbn [static_values] in Hs.
    rewrite lookup_app in Hs.
    destruct (lookup s (live_locals (rs_frames rs))) as [local | ]
      eqn:Hlocal_s.
    + injection Hs as ->. split.
      * unfold live, value_within. cbn [value_all].
        unfold symbol_within, local_in. rewrite Hlocal_s. apply orb_true_r.
      * intros Hnot. exfalso.
        pose proof (forallb_in _ _ (s, allocated) Hlocal
                      (lookup_in _ _ _ Hlocal_s))
          as Hfit.
        cbv beta iota in Hfit.
        assert (Hmay : closure_may_be_local cert id = true).
        { destruct allocated
            as [code slots0 | slot symbols decls slots0 | | | | | ];
            try discriminate Hof.
          - injection Hof as <- _. exact Hfit.
          - destruct (lookup slot decls) as [code | ] eqn:Hslot;
              [ | discriminate Hof ].
            injection Hof as <- _.
            exact (forallb_in _ _ (slot, code) Hfit (lookup_in _ _ _ Hslot)). }
        unfold closure_may_be_local in Hmay. rewrite Hentry in Hmay.
        congruence.
    + assert (Hheap : heap_symbol rs s = true).
      { unfold heap_symbol. rewrite Hs. destruct allocated; apply orb_true_r. }
      split; [ | intros _; exact Hheap ].
      unfold live, value_within. cbn [value_all]. unfold symbol_within.
      rewrite Hheap. reflexivity.
Qed.

(* The fields of a tuple that is live are live; those of one that
   reaches symbols of the heap only reach such symbols only. *)
Lemma tuple_placed (tuple : value) (size : nat) (fields : list value) :
  live rs tuple ->
  tuple_fields (live_statics rs) size tuple = Some fields ->
  Forall (live rs) fields
  ∧ (value_all (heap_symbol rs) tuple = true ->
     Forall (fun v => value_all (heap_symbol rs) v = true) fields).
Proof.
  intros Hlive Hfields.
  destruct (tuple_fields_block _ _ _ _ Hfields) as [-> | Hblock];
    [ split; [ constructor | intros _; constructor ] | ].
  destruct (FlambdaSemanticsProofs.block_of_cases _ _ _ _ Hblock)
    as [-> | (s & -> & Hlookup)].
  - split.
    + unfold live, value_within in Hlive. cbn [value_all] in Hlive.
      exact (proj1 (forallb_Forall _ _) Hlive).
    + intros Hheap. cbn [value_all] in Hheap.
      exact (proj1 (forallb_Forall _ _) Hheap).
  - unfold live_statics in Hlookup. cbn [static_values] in Hlookup.
    rewrite lookup_app in Hlookup.
    destruct (lookup s (live_locals (rs_frames rs))) as [local | ]
      eqn:Hlocal_s.
    + injection Hlookup as ->.
      pose proof (forallb_in _ _ (s, Static_block 0 fields)
                    (frames_within [] (rs_frames rs) eq_refl Hdeep)
                    (lookup_in _ _ _ Hlocal_s))
        as Hall.
      cbv beta iota in Hall. cbn [allocated_all] in Hall.
      split; [ exact (proj1 (forallb_Forall _ _) Hall) | ].
      intros Hheap. cbn [value_all] in Hheap.
      pose proof (heap_not_local rs s Hfresh Hheap) as Hnot.
      unfold not_local_symbol, value_local in Hnot. rewrite Hlocal_s in Hnot.
      cbn [is_some negb] in Hnot. rewrite andb_false_r in Hnot. discriminate.
    + pose proof (forallb_in _ _ (s, Static_block 0 fields) Hstatics
                    (lookup_in _ _ _ Hlookup))
        as Hall.
      cbv beta iota in Hall. cbn [allocated_all] in Hall.
      split; [ | intros _; exact (proj1 (forallb_Forall _ _) Hall) ].
      apply Forall_forall. intros w Hin.
      exact (heap_within rs None w (forallb_in _ _ w Hall Hin)).
Qed.

End Reads.

(* ======================================================================== *)
(* Lifetimes of values                                                      *)
(* ======================================================================== *)

(* The place of a region with a bound of the newest, from its place
   with none. *)
Lemma region_place_above (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable),
    region_place opens ms above r
    = match region_place opens ms None r with
      | Some (m, None) => Some (m, above)
      | other => other
      end.
Proof.
  induction opens as [ | x opens IH]; intros ms above r; [ reflexivity | ].
  cbn [region_place]. destruct ms as [ | m ms]; [ reflexivity | ].
  destruct (Pos.eqb x r); [ reflexivity | ].
  rewrite (IH ms (Some m) r).
  destruct (region_place opens ms None r) as [[m' [b | ]] | ]; reflexivity.
Qed.

(* A value as a lifetime says with a bound of the newest region is so
   with none. *)
Lemma holds_none (rs : region_state) (opens : list variable) (ms : list mark)
    (above : option symbol) (l : lifetime) (v : value) :
  lifetime_holds rs opens ms above l v -> lifetime_holds rs opens ms None l v.
Proof.
  destruct l as [ | | r | ]; cbn [lifetime_holds]; try (intros H; exact H).
  - apply within_later. unfold caller_bound.
    destruct (lowest ms) as [b | ]; [ | discriminate ].
    intros b' [= <-]. exists b. split; [ reflexivity | lia ].
  - rewrite (region_place_above opens ms above r).
    destruct (region_place opens ms None r) as [[m [b | ]] | ];
      try (intros H; exact H).
    intros H Hstack.
    apply (within_later rs above None v); [ | exact (H Hstack) ].
    intros b' H'. discriminate H'.
Qed.

(* A value that reaches symbols of the heap only is as any lifetime
   says. *)
Lemma heap_holds (rs : region_state) (opens : list variable) (ms : list mark)
    (above : option symbol) (l : lifetime) (v : value) :
  value_all (heap_symbol rs) v = true -> lifetime_holds rs opens ms above l v.
Proof.
  intros Hheap. destruct l as [ | | r | ]; cbn [lifetime_holds].
  - exact Hheap.
  - exact (heap_within rs _ v Hheap).
  - destruct (region_place opens ms above r) as [[m stop] | ]; [ | exact I ].
    intros _. exact (heap_within rs _ v Hheap).
  - exact I.
Qed.

(* A partial application is as a lifetime says when its callee and its
   arguments are, and conversely. *)
Lemma holds_partial (rs : region_state) (opens : list variable)
    (ms : list mark) (above : option symbol) (l : lifetime) (callee : value)
    (args : list value) (args_arity : arity) :
  lifetime_holds rs opens ms above l
    (Value_partial_application callee args args_arity)
  <-> lifetime_holds rs opens ms above l callee
      ∧ Forall (lifetime_holds rs opens ms above l) args.
Proof.
  assert (Hall :
    ∀ ok : symbol -> bool,
      value_all ok (Value_partial_application callee args args_arity) = true
      <-> value_all ok callee = true
          ∧ Forall (fun v => value_all ok v = true) args).
  { intros ok. cbn [value_all]. rewrite andb_true_iff, forallb_Forall.
    reflexivity. }
  destruct l as [ | | r | ]; cbn [lifetime_holds].
  - apply Hall.
  - apply Hall.
  - destruct (region_place opens ms above r) as [[m stop] | ] eqn:Hplace.
    + split.
      * intros H. split.
        -- intros Hstack. exact (proj1 (proj1 (Hall _) (H Hstack))).
        -- apply Forall_forall. intros arg Hin. cbn [lifetime_holds].
           rewrite Hplace. intros Hstack.
           exact (proj1 (Forall_forall _ _) (proj2 (proj1 (Hall _) (H Hstack)))
                    arg Hin).
      * intros [Hcallee Hargs] Hstack. apply Hall.
        split; [ exact (Hcallee Hstack) | ].
        apply Forall_forall. intros arg Hin.
        pose proof (proj1 (Forall_forall _ _) Hargs arg Hin) as H0.
        cbn [lifetime_holds] in H0. rewrite Hplace in H0. exact (H0 Hstack).
    + split; [ | intros _; exact I ].
      intros _. split; [ exact I | ]. apply Forall_forall. intros arg _.
      cbn [lifetime_holds]. rewrite Hplace. exact I.
  - split; [ | intros _; exact I ].
    intros _. split; [ exact I | ]. apply Forall_forall. intros arg _.
    exact I.
Qed.

(* ======================================================================== *)
(* Arguments against the flags of the parameters                            *)
(* ======================================================================== *)

(* Each argument at a parameter reaches symbols of the heap only, or
   the parameter may be given a local value. *)
Definition fit_pointwise (rs : region_state) (flags : list bool)
    (vs : list value) : Prop :=
  ∀ (i : nat) (v : value) (flag : bool),
    nth_error vs i = Some v ->
    nth_error flags i = Some flag ->
    value_all (heap_symbol rs) v = true ∨ flag = true.

Lemma pointwise_values_fit (rs : region_state) (flags : list bool) :
  ∀ vs : list value,
    length vs <= length flags ->
    fit_pointwise rs flags vs ->
    values_fit rs flags vs = true.
Proof.
  induction flags as [ | flag flags IH]; intros [ | v vs];
    cbn [values_fit length]; try reflexivity; try lia.
  intros Hlength Hfit. apply andb_true_iff. split.
  - destruct (Hfit 0 v flag eq_refl eq_refl) as [-> | ->];
      [ reflexivity | apply orb_true_r ].
  - apply IH; [ lia | ]. intros i w flag0 Hw Hflag.
    exact (Hfit (S i) w flag0 Hw Hflag).
Qed.

Lemma nth_error_firstn_some {A : Type} (l : list A) :
  ∀ (n i : nat) (a : A),
    nth_error (firstn n l) i = Some a -> nth_error l i = Some a.
Proof.
  induction l as [ | a0 l IH]; intros [ | n] [ | i] a; cbn [firstn nth_error];
    try discriminate; [ intros H; exact H | exact (IH n i a) ].
Qed.

Lemma fit_pointwise_firstn (rs : region_state) (flags : list bool) (n : nat)
    (vs : list value) :
  fit_pointwise rs flags vs -> fit_pointwise rs flags (firstn n vs).
Proof.
  intros Hfit i v flag Hv.
  exact (Hfit i v flag (nth_error_firstn_some vs n i v Hv)).
Qed.

Lemma bind_params_same_length (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    bind_params params vs vars = Some vars' -> length vs = length params.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars vars';
    cbn [bind_params length]; try discriminate; [ reflexivity | ].
  intros H. rewrite (IH vs _ vars' H). reflexivity.
Qed.

Lemma forallb_id_nth (flags : list bool) (i : nat) (flag : bool) :
  forallb (fun flag => flag) flags = true ->
  nth_error flags i = Some flag ->
  flag = true.
Proof.
  intros Hall Hflag.
  exact (forallb_in _ _ flag Hall (nth_error_In flags i Hflag)).
Qed.

(* ======================================================================== *)
(* The code and the certificate                                             *)
(* ======================================================================== *)

Lemma check_code_parts (cert : certificate) (codes : list (code_id * code))
    (id : code_id) (c : code) (entry : code_entry) :
  check_code cert codes id c = true ->
  lookup id (cert_code cert) = Some entry ->
  length (ce_params_local entry) = length (code_params c)
  ∧ length (code_params c) = cardinal_unarized (code_params_arity c)
  ∧ (cert_indirect_local cert = true ->
     forallb (fun flag => flag) (ce_params_local entry) = true)
  ∧ (code_is_tupled c = true ->
     forallb (fun flag => flag) (ce_params_local entry) = true
     ∨ forallb negb (ce_params_local entry) = true)
  ∧ (ce_returns_local entry = true ->
     cert_unknown_returns_local cert = true).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [check_code code_params code_params_arity code_is_tupled].
  intros H Hentry. rewrite Hentry in H.
  apply andb_true_iff in H as [H _].
  apply andb_true_iff in H as [H Hreturns].
  apply andb_true_iff in H as [H Htupled].
  apply andb_true_iff in H as [H Hindirect].
  apply andb_true_iff in H as [H _].
  apply andb_true_iff in H as [H _].
  apply andb_true_iff in H as [Hlen1 Hlen2].
  apply Nat.eqb_eq in Hlen1, Hlen2.
  split; [ exact Hlen1 | ]. split; [ exact Hlen2 | ]. split; [ | split ].
  - intros Hi. rewrite Hi in Hindirect. exact Hindirect.
  - intros Ht. rewrite Ht in Htupled. cbn [negb orb] in Htupled.
    apply orb_true_iff in Htupled. exact Htupled.
  - intros Hr. rewrite Hr in Hreturns. exact Hreturns.
Qed.

Lemma check_code_entry (cert : certificate) (codes : list (code_id * code))
    (id : code_id) (c : code) :
  check_code cert codes id c = true ->
  ∃ entry : code_entry, lookup id (cert_code cert) = Some entry.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [check_code]. destruct (lookup id (cert_code cert)) as [entry | ];
    [ intros _; exists entry; reflexivity | discriminate ].
Qed.

Lemma full_arity_le_max (codes : list (code_id * code)) (id : code_id)
    (c : code) :
  In (id, c) codes -> full_arity c <= max_full_arity codes.
Proof.
  unfold max_full_arity.
  induction codes as [ | [id0 c0] codes IH]; [ contradiction | ].
  cbn [fold_right]. intros [[= <- <-] | Hin]; [ lia | ].
  specialize (IH Hin). lia.
Qed.

Lemma full_arity_le_max_local (cert : certificate)
    (codes : list (code_id * code)) (id : code_id) (c : code)
    (entry : code_entry) :
  In (id, c) codes ->
  lookup id (cert_code cert) = Some entry ->
  ce_closure_local entry = true ->
  full_arity c <= max_local_closure_arity cert codes.
Proof.
  intros Hin Hentry Hlocal. unfold max_local_closure_arity.
  induction codes as [ | [id0 c0] codes IH]; [ contradiction | ].
  cbn [fold_right]. destruct Hin as [[= <- <-] | Hin].
  - rewrite Hentry, Hlocal. lia.
  - specialize (IH Hin).
    destruct (lookup id0 (cert_code cert)) as [entry0 | ]; [ | exact IH ].
    destruct (ce_closure_local entry0); lia.
Qed.

Lemma later_call_code (codes : list (code_id * code)) (id : code_id)
    (c : code) (count : nat) :
  In (id, c) codes -> full_arity c < count -> later_call codes count = true.
Proof.
  intros Hin Hlt. unfold later_call. apply orb_true_iff. left.
  apply existsb_exists. exists (id, c).
  split; [ exact Hin | apply Nat.ltb_lt; exact Hlt ].
Qed.

Lemma later_call_wide (codes : list (code_id * code)) (count : nat) :
  1 < max_full_arity codes -> 1 < count -> later_call codes count = true.
Proof.
  intros Hwide Hcount. unfold later_call. apply orb_true_iff. right.
  apply andb_true_iff. split; apply Nat.ltb_lt; assumption.
Qed.

(* ======================================================================== *)
(* The call of a closure, by its shape                                      *)
(* ======================================================================== *)

Section Call.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

Hypothesis Hfresh : symbols_fresh rs = true.

Hypothesis Hcode : code_checked cert codes rs.

Hypothesis Hlocal : local_closures_fit cert rs = true.

Hypothesis Hdeep : frames_deep rs (rs_frames rs) = true.

Hypothesis Hstatics : statics_deep rs = true.

Hypothesis Hgood_statics :
  Forall (fun binding => allocated_values (good cert codes rs) (snd binding))
    (static_values (live_statics rs)).

Variable hp : heap.

(* The code that a closure is of is code that the check was given,
   and has an entry. *)
Lemma called_code (id : code_id) (c : code) :
  lookup id (static_code (live_statics rs)) = Some c ->
  lookup id codes = Some c
  ∧ check_code cert codes id c = true
  ∧ ∃ entry : code_entry, lookup id (cert_code cert) = Some entry.
Proof.
  intros Hc. destruct Hcode as [_ Hcodes].
  destruct (Hcodes id c Hc) as [Hin Hcheck].
  split; [ exact Hin | ]. split; [ exact Hcheck | ].
  exact (check_code_entry cert codes id c Hcheck).
Qed.

(* The call of a closure of code, the arguments being as the code
   takes them: the three shapes, each with what the caller of this
   lemma owes. *)
Lemma call_closure (a : caller) (closure : value) (id : code_id)
    (slots : list (value_slot * value)) (c : code) (entry : code_entry)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (shape : call_shape) (s' : state) :
  caller_ok rs a ->
  closure_of (live_statics rs) closure = Some (id, slots) ->
  lookup id (static_code (live_statics rs)) = Some c ->
  lookup id (cert_code cert) = Some entry ->
  good cert codes rs closure ->
  Forall (good cert codes rs) vs ->
  Forall (live rs) vs ->
  fit_pointwise rs (ce_params_local entry) vs ->
  raise_ok cert codes rs a exn_to ->
  cont_values_all (good cert codes rs) return_to ->
  cont_values_all (good cert codes rs) exn_to ->
  shape_of_code_call closure c vs args_arity return_arity = Some shape ->
  (length args_arity = length (code_params_arity c) ->
   ret_ok cert codes rs a
     (if ce_returns_local entry then newest_lifetime (caller_check a)
      else Forever)
     return_to) ->
  (length (code_params_arity c) < length args_arity ->
   map kind_of_value vs = unarize args_arity ->
   unarize_per_parameter (firstn (length (code_params_arity c)) args_arity)
   = unarize_per_parameter (code_params_arity c) ->
   ret_ok cert codes rs a
     (if ce_returns_local entry then newest_lifetime (caller_check a)
      else Forever)
     (Over_application (skipn (cardinal_unarized (code_params_arity c)) vs)
        (partially_apply args_arity (length (code_params_arity c)))
        return_arity return_to exn_to)) ->
  (length args_arity < length (code_params_arity c) ->
   code_is_tupled c = false ->
   map kind_of_value vs = unarize args_arity ->
   partial_or_over_application args_arity return_arity (code_params_arity c)
     (code_result_arity c)
   = true ->
   ∀ s'' : state,
     apply_cont_value return_to
       [Value_partial_application closure vs args_arity] (live_statics rs) hp
     = Step s'' ->
     entered cert codes rs s'') ->
  run_call
    (fun partial => apply_cont_value return_to [partial] (live_statics rs) hp)
    (Some shape) return_arity return_to exn_to (live_statics rs) hp
  = Step s' ->
  entered cert codes rs s'.
Proof.
  intros Hcaller Hof Hc Hentry Hgood_closure Hgood_vs Hlive_vs Hfit Hraise
    Hgood_ret Hgood_exn Hshape Kfull Kover Kpartial.
  destruct (closure_place cert codes rs Hlocal closure id slots entry Hof
              Hgood_closure Hentry)
    as [Hlive_closure Hheap_closure].
  destruct (called_code id c Hc) as (_ & Hcheck & _).
  destruct (check_code_parts cert codes id c entry Hcheck Hentry)
    as (Hlen_flags & Hlen_params & _).
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hshape)
    as [[Hlen ->]
       | [(Hlen & Hkinds & Hpo & ->) | (Hlen & Hkinds & Hpo & Htupled & ->)]];
    cbn [run_call].
  - intros Hcall.
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ Hcall)
      as (callee_vars & _ & _ & _ & _ & Hbind & _).
    apply (enter_code cert codes rs a closure id c entry vs args_arity
             return_arity return_to exn_to hp s' Hcaller Hcheck Hentry
             Hgood_closure Hlive_closure Hheap_closure Hgood_vs Hlive_vs);
      [ | exact (Kfull Hlen) | exact Hraise | exact Hgood_ret | exact Hgood_exn
        | exact Hcall ].
    apply pointwise_values_fit; [ | exact Hfit ].
    rewrite (bind_params_same_length _ _ _ _ Hbind), Hlen_flags. lia.
  - intros Hcall.
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ Hcall)
      as (callee_vars & _ & _ & Hper & _ & Hbind & _).
    rewrite <- (firstn_skipn (cardinal_unarized (code_params_arity c)) vs)
      in Hgood_vs, Hlive_vs.
    apply Forall_app in Hgood_vs as [Hgood_first Hgood_rest].
    apply Forall_app in Hlive_vs as [Hlive_first _].
    apply (enter_code cert codes rs a closure id c entry
             (firstn (cardinal_unarized (code_params_arity c)) vs)
             (firstn (length (code_params_arity c)) args_arity)
             (code_result_arity c)
             (Over_application
                (skipn (cardinal_unarized (code_params_arity c)) vs)
                (partially_apply args_arity (length (code_params_arity c)))
                return_arity return_to exn_to)
             exn_to hp s' Hcaller Hcheck Hentry Hgood_closure Hlive_closure
             Hheap_closure Hgood_first Hlive_first);
      [ | exact (Kover Hlen Hkinds Hper) | exact Hraise | | exact Hgood_exn
        | exact Hcall ].
    + apply pointwise_values_fit;
        [ | apply fit_pointwise_firstn; exact Hfit ].
      rewrite (bind_params_same_length _ _ _ _ Hbind), Hlen_flags. lia.
    + split; [ exact Hgood_rest | ].
      split; [ exact Hgood_ret | exact Hgood_exn ].
  - intros Hcall. exact (Kpartial Hlen Htupled Hkinds Hpo s' Hcall).
Qed.

(* What a continuation that takes a value of a lifetime of a caller
   does with values of that lifetime: the conclusion of the induction
   on [ret_ok], as the calls use it. *)
Definition returns_ok (a : caller) (l : lifetime) (c : cont_value) : Prop :=
  ∀ (vs : list value) (s'' : state),
    Forall (good cert codes rs) vs ->
    Forall (live rs) vs ->
    Forall
      (lifetime_holds rs (cs_open (caller_check a)) (caller_marks a) None l)
      vs ->
    apply_cont_value c vs (live_statics rs) hp = Step s'' ->
    entered cert codes rs s''.

Lemma caller_marked (a : caller) :
  caller_ok rs a ->
  Forall2 (marked (caller_vars a)) (cs_open (caller_check a))
    (caller_marks a).
Proof. intros [H _ _ _ _]. exact (proj1 (open_marks_marked _ _ _) H). Qed.

(* A value that the caller holds since the call. *)
Lemma caller_holds_facts (a : caller) (l : lifetime) (v : value) :
  caller_ok rs a ->
  caller_holds rs a l v ->
  live rs v
  ∧ lifetime_holds rs (cs_open (caller_check a)) (caller_marks a) None l v
  ∧ (is_forever l = true -> value_all (heap_symbol rs) v = true).
Proof.
  intros Hcaller [Husable Hholds].
  pose proof (holds_none _ _ _ _ _ _ Hholds) as Hnone.
  split; [ | split; [ exact Hnone | ] ].
  - apply (holds_live rs (caller_vars a) (cs_open (caller_check a))
             (caller_marks a) l v (caller_marked a Hcaller)
             (caller_placed rs a Hcaller));
      [ | exact Hnone ].
    apply usable_rank. exact Husable.
  - destruct l; cbn [is_forever]; try discriminate. intros _. exact Hnone.
Qed.

Lemma Forall2_skipn {A B : Type} (R : A -> B -> Prop) (l : list A)
    (l' : list B) :
  Forall2 R l l' -> ∀ n : nat, Forall2 R (skipn n l) (skipn n l').
Proof.
  induction 1 as [ | a b l l' Hab Hrest IH]; intros [ | n]; cbn [skipn];
    try constructor; try assumption.
  exact (IH n).
Qed.

Lemma In_skipn {A : Type} (l : list A) :
  ∀ (n : nat) (a : A), In a (skipn n l) -> In a l.
Proof.
  induction l as [ | a0 l IH]; intros [ | n] a; cbn [skipn];
    try (intros H; exact H).
  intros H. right. exact (IH n a H).
Qed.

(* The tests of [unknown_call]. *)
Lemma unknown_call_fit (cs : check_state) (callee : lifetime)
    (args : list lifetime) (count : nat) (l : lifetime) :
  unknown_call cert codes cs callee args count = Some l ->
  forallb is_forever args = true ∨ cert_indirect_local cert = true.
Proof.
  unfold unknown_call. cbv zeta.
  destruct (forallb is_forever args); [ intros _; left; reflexivity | ].
  destruct (cert_indirect_local cert); [ intros _; right; reflexivity | ].
  discriminate.
Qed.

Lemma unknown_call_local (cs : check_state) (callee : lifetime)
    (args : list lifetime) (count : nat) (l : lifetime) :
  cert_partial_local cert = true ->
  unknown_call cert codes cs callee args count = Some l ->
  l
  = if may_be_partial codes count
    then
      shorter (cs_open cs) (unknown_lifetime cert cs)
        (shortest cs (callee :: args))
    else unknown_lifetime cert cs.
Proof.
  intros Hlocal_cert. unfold unknown_call. cbv zeta. rewrite Hlocal_cert.
  destruct (forallb is_forever args || cert_indirect_local cert);
    [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

Lemma unknown_call_strict (cs : check_state) (callee : lifetime)
    (args : list lifetime) (count : nat) (l : lifetime) :
  cert_partial_local cert = false ->
  unknown_call cert codes cs callee args count = Some l ->
  l = unknown_lifetime cert cs
  ∧ (may_be_partial codes count = true -> forallb is_forever args = true)
  ∧ (is_forever callee = false ->
     max_local_closure_arity cert codes <= count).
Proof.
  intros Hlocal_cert. unfold unknown_call. cbv zeta. rewrite Hlocal_cert.
  fold (may_be_partial codes count). fold (unknown_lifetime cert cs).
  destruct (forallb is_forever args || cert_indirect_local cert);
    [ | discriminate ].
  destruct (negb (forallb is_forever args) && may_be_partial codes count)
    eqn:Hfirst; [ discriminate | ].
  cbn [orb].
  destruct (negb (is_forever callee)
            && (count <? max_local_closure_arity cert codes))
    eqn:Hsecond; [ discriminate | ].
  cbn [orb].
  destruct (later_call codes count
            && negb (is_forever (unknown_lifetime cert cs))
            && (1 <? max_local_closure_arity cert codes));
    [ discriminate | ].
  intros [= <-].
  split; [ reflexivity | ]. split.
  - intros Hpartial. rewrite Hpartial, andb_true_r in Hfirst.
    apply negb_false_iff in Hfirst. exact Hfirst.
  - intros Hcallee. rewrite Hcallee in Hsecond. cbn [negb andb] in Hsecond.
    apply Nat.ltb_ge in Hsecond. exact Hsecond.
Qed.

(* The call of a closure by a call whose callee is not known. *)
Lemma call_unknown_closure (a : caller) (l_c : lifetime) (ls : list lifetime)
    (count : nat) (l' : lifetime) (closure : value) (id : code_id)
    (slots : list (value_slot * value)) (c : code) (entry : code_entry)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (shape : call_shape) (s' : state) :
  caller_ok rs a ->
  unknown_call cert codes (caller_check a) l_c ls count = Some l' ->
  ret_ok cert codes rs a l' return_to ->
  raise_ok cert codes rs a exn_to ->
  returns_ok a l' return_to ->
  cont_values_all (good cert codes rs) return_to ->
  cont_values_all (good cert codes rs) exn_to ->
  closure_of (live_statics rs) closure = Some (id, slots) ->
  lookup id (static_code (live_statics rs)) = Some c ->
  lookup id (cert_code cert) = Some entry ->
  good cert codes rs closure ->
  Forall (good cert codes rs) vs ->
  Forall (live rs) vs ->
  fit_pointwise rs (ce_params_local entry) vs ->
  (length (code_params_arity c) < length args_arity ->
   map kind_of_value vs = unarize args_arity ->
   unarize_per_parameter (firstn (length (code_params_arity c)) args_arity)
   = unarize_per_parameter (code_params_arity c) ->
   later_call codes count = true
   ∧ ∃ ls_rem : list lifetime,
       Forall2 (caller_holds rs a) ls_rem
         (skipn (cardinal_unarized (code_params_arity c)) vs)
       ∧ ∀ l0 : lifetime, In l0 ls_rem -> In l0 ls) ->
  (length args_arity < length (code_params_arity c) ->
   code_is_tupled c = false ->
   map kind_of_value vs = unarize args_arity ->
   count < max_full_arity codes
   ∧ lifetime_holds rs (cs_open (caller_check a)) (caller_marks a) None
       (shortest (caller_check a) (l_c :: ls)) closure
   ∧ Forall
       (lifetime_holds rs (cs_open (caller_check a)) (caller_marks a) None
          (shortest (caller_check a) (l_c :: ls)))
       vs
   ∧ (cert_partial_local cert = false ->
      value_all (heap_symbol rs) closure = true
      ∧ Forall (fun v => value_all (heap_symbol rs) v = true) vs)) ->
  shape_of_code_call closure c vs args_arity return_arity = Some shape ->
  run_call
    (fun partial => apply_cont_value return_to [partial] (live_statics rs) hp)
    (Some shape) return_arity return_to exn_to (live_statics rs) hp
  = Step s' ->
  entered cert codes rs s'.
Proof.
  intros Hcaller Hunknown Hret Hraise Hreturns Hgood_ret Hgood_exn Hof Hc
    Hentry Hgood_closure Hgood_vs Hlive_vs Hfit Hover Hpartial Hshape.
  destruct (called_code id c Hc) as (Hin_codes & Hcheck & _).
  destruct (check_code_parts cert codes id c entry Hcheck Hentry)
    as (Hlen_flags & Hlen_params & _ & _ & Hreturns_local).
  set (cs := caller_check a) in *.
  set (first :=
    if ce_returns_local entry then newest_lifetime cs else Forever).
  assert (Hfirst : first = Forever ∨ first = unknown_lifetime cert cs).
  { unfold first, unknown_lifetime.
    destruct (ce_returns_local entry); [ | left; reflexivity ].
    rewrite (Hreturns_local eq_refl). right. reflexivity. }
  assert (Hfirst_rank :
    rank (cs_open cs) (unknown_lifetime cert cs) <= rank (cs_open cs) first).
  { destruct Hfirst as [-> | ->]; [ | lia ].
    pose proof (rank_le (cs_open cs) (unknown_lifetime cert cs)).
    cbn [rank]. lia. }
  apply (call_closure a closure id slots c entry vs args_arity return_arity
           return_to exn_to shape s' Hcaller Hof Hc Hentry Hgood_closure
           Hgood_vs Hlive_vs Hfit Hraise Hgood_ret Hgood_exn Hshape).
  - intros _. apply (ret_longer cert codes rs a first l' return_to Hret).
    exact
      (Nat.le_trans _ _ _
         (unknown_call_le cert codes cs l_c ls count l' Hunknown) Hfirst_rank).
  - intros Hlen Hkinds Hper.
    destruct (Hover Hlen Hkinds Hper)
      as (Hlater & ls_rem & Hholds_rem & Hincl).
    assert (Hcount' :
      1 <= length
             (partially_apply args_arity (length (code_params_arity c)))).
    { unfold partially_apply. rewrite length_skipn. lia. }
    destruct (unknown_call_later cert codes cs l_c first ls ls_rem count _ l'
                Hunknown Hlater Hcount' Hincl Hfirst)
      as (l2 & Hcall2 & Hrank).
    exact
      (Ret_over cert codes rs a first l2 ls_rem _ _ return_arity return_to
         exn_to Hholds_rem Hcall2
         (ret_longer cert codes rs a l2 l' return_to Hret Hrank) Hraise).
  - intros Hlen Htupled Hkinds Hpo s'' Hcall.
    destruct (Hpartial Hlen Htupled Hkinds)
      as (Hcount & Hkept_closure & Hkept_vs & Hstrict).
    assert (Hmay : may_be_partial codes count = true).
    { unfold may_be_partial. apply orb_true_iff. left. apply Nat.ltb_lt.
      exact Hcount. }
    destruct (closure_place cert codes rs Hlocal closure id slots entry Hof
                Hgood_closure Hentry)
      as [Hlive_closure _].
    assert (Hlength : length vs <= length (ce_params_local entry)).
    { rewrite Hlen_flags, Hlen_params.
      rewrite <- (length_map kind_of_value vs), Hkinds.
      unfold partial_or_over_application in Hpo.
      apply andb_true_iff in Hpo as [Hagree _].
      apply (prefix_cardinal _ _ Hagree). lia. }
    assert (Hpap_fits :
      pap_fits cert codes rs closure vs args_arity = true).
    { unfold pap_fits. rewrite Hof, Hin_codes, Hentry, Htupled.
      cbn [negb andb].
      assert (H : (length args_arity <? length (code_params_arity c)) = true)
        by (apply Nat.ltb_lt; exact Hlen).
      rewrite H. cbn [andb].
      destruct (cert_partial_local cert) eqn:Hlocal_cert.
      - rewrite (pointwise_values_fit rs _ vs Hlength Hfit). apply orb_true_r.
      - destruct (Hstrict eq_refl) as [Hheap_closure Hheap_vs].
        rewrite Hheap_closure, (proj2 (forallb_Forall _ _) Hheap_vs).
        reflexivity. }
    apply (Hreturns [Value_partial_application closure vs args_arity] s'');
      [ | | | exact Hcall ].
    + apply Forall_cons; [ | apply Forall_nil ].
      destruct Hgood_closure as [Hplain_closure Hfits_closure].
      split.
      * cbn [value_plain]. rewrite Hplain_closure. cbn [andb].
        apply forallb_forall. intros v Hin.
        exact (proj1 (proj1 (Forall_forall _ _) Hgood_vs v Hin)).
      * cbn [value_paps]. rewrite Hpap_fits, Hfits_closure. cbn [andb].
        apply forallb_forall. intros v Hin.
        exact (proj2 (proj1 (Forall_forall _ _) Hgood_vs v Hin)).
    + apply Forall_cons; [ | apply Forall_nil ].
      apply live_partial. split; assumption.
    + apply Forall_cons; [ | apply Forall_nil ].
      destruct (cert_partial_local cert) eqn:Hlocal_cert.
      * rewrite (unknown_call_local cs l_c ls count l' Hlocal_cert Hunknown),
          Hmay.
        assert (Hrank :
          rank (cs_open cs)
            (shorter (cs_open cs) (unknown_lifetime cert cs)
               (shortest cs (l_c :: ls)))
          <= rank (cs_open cs) (shortest cs (l_c :: ls)))
          by (rewrite rank_shorter; lia).
        apply holds_partial. split.
        -- exact (holds_rank rs (caller_vars a) _ _ _ _ closure
                    (caller_marked a Hcaller) (caller_placed rs a Hcaller)
                    Hrank Hkept_closure).
        -- revert Hkept_vs. apply Forall_impl. intros v.
           exact (holds_rank rs (caller_vars a) _ _ _ _ v
                    (caller_marked a Hcaller) (caller_placed rs a Hcaller)
                    Hrank).
      * destruct (Hstrict eq_refl) as [Hheap_closure Hheap_vs].
        apply heap_holds. cbn [value_all]. rewrite Hheap_closure.
        exact (proj2 (forallb_Forall _ _) Hheap_vs).
Qed.

Lemma In_firstn {A : Type} (l : list A) :
  ∀ (n : nat) (a : A), In a (firstn n l) -> In a l.
Proof.
  induction l as [ | a0 l IH]; intros [ | n] a; cbn [firstn];
    try contradiction.
  intros [<- | H]; [ left; reflexivity | right; exact (IH n a H) ].
Qed.

Lemma singletons_cardinal (l : arity) :
  (∀ component : arity_component,
     In component l -> ∃ kind : flambda_kind, component = Singleton kind) ->
  length (concat (unarize_per_parameter l)) = length l.
Proof.
  induction l as [ | component l IH]; intros Hall; [ reflexivity | ].
  cbn [unarize_per_parameter map concat length]. rewrite length_app.
  destruct (Hall component (or_introl eq_refl)) as [kind ->].
  cbn [unarize_component length].
  change (map unarize_component l) with (unarize_per_parameter l).
  rewrite (IH (fun component0 Hin => Hall component0 (or_intror Hin))).
  reflexivity.
Qed.

Lemma Forall2_right_in {A B : Type} (R : A -> B -> Prop) (Q : B -> Prop)
    (l : list A) (l' : list B) :
  Forall2 R l l' ->
  (∀ (a : A) (b : B), In a l -> R a b -> Q b) ->
  Forall Q l'.
Proof.
  induction 1 as [ | a b l l' Hab _ IH]; intros Himpl; constructor.
  - exact (Himpl a b (or_introl eq_refl) Hab).
  - apply IH. intros a0 b0 Hin. exact (Himpl a0 b0 (or_intror Hin)).
Qed.

(* A call whose callee is not known. *)
Lemma call_unknown (a : caller) (callee : value) (l_c : lifetime)
    (ls : list lifetime) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (l' : lifetime) (s' : state) :
  caller_ok rs a ->
  good cert codes rs callee ->
  live rs callee ->
  lifetime_holds rs (cs_open (caller_check a)) (caller_marks a) None l_c
    callee ->
  Forall (good cert codes rs) vs ->
  Forall2 (caller_holds rs a) ls vs ->
  unknown_call cert codes (caller_check a) l_c ls (length args_arity)
  = Some l' ->
  ret_ok cert codes rs a l' return_to ->
  raise_ok cert codes rs a exn_to ->
  returns_ok a l' return_to ->
  cont_values_all (good cert codes rs) return_to ->
  cont_values_all (good cert codes rs) exn_to ->
  call_function callee Indirect vs args_arity return_arity return_to exn_to
    (live_statics rs) hp
  = Step s' ->
  entered cert codes rs s'.
Proof.
  intros Hcaller Hgood_callee Hlive_callee Hholds_callee Hgood_vs Hholds
    Hunknown Hret Hraise Hreturns Hgood_ret Hgood_exn.
  unfold call_function.
  destruct (shape_of_call (live_statics rs) callee Indirect vs args_arity
              return_arity)
    as [shape | ] eqn:Hshape; [ | discriminate ].
  set (cs := caller_check a) in *.
  set (kept := shortest cs (l_c :: ls)).
  assert (Hfacts :
    Forall2
      (fun l v =>
         live rs v
         ∧ lifetime_holds rs (cs_open cs) (caller_marks a) None l v
         ∧ (is_forever l = true -> value_all (heap_symbol rs) v = true))
      ls vs).
  { revert Hholds. apply Forall2_impl. intros l v.
    exact (caller_holds_facts a l v Hcaller). }
  assert (Hlive_vs : Forall (live rs) vs).
  { apply (Forall2_right_in _ _ ls vs Hfacts). intros l v _ H.
    exact (proj1 H). }
  assert (Hheap_vs :
    forallb is_forever ls = true ->
    Forall (fun v => value_all (heap_symbol rs) v = true) vs).
  { intros Hforever. apply (Forall2_right_in _ _ ls vs Hfacts).
    intros l v Hin H. exact (proj2 (proj2 H) (forallb_in _ _ l Hforever Hin)). }
  assert (Hkept_vs :
    Forall (lifetime_holds rs (cs_open cs) (caller_marks a) None kept) vs).
  { apply (Forall2_right_in _ _ ls vs Hfacts). intros l v Hin H.
    apply (holds_rank rs (caller_vars a) _ _ l kept v (caller_marked a Hcaller)
             (caller_placed rs a Hcaller));
      [ | exact (proj1 (proj2 H)) ].
    exact (rank_shortest_le cs (l_c :: ls) l (or_intror Hin)). }
  assert (Hkept_callee :
    ∀ w : value,
      lifetime_holds rs (cs_open cs) (caller_marks a) None l_c w ->
      lifetime_holds rs (cs_open cs) (caller_marks a) None kept w).
  { intros w.
    apply (holds_rank rs (caller_vars a) _ _ l_c kept w
             (caller_marked a Hcaller) (caller_placed rs a Hcaller)).
    exact (rank_shortest_le cs (l_c :: ls) l_c (or_introl eq_refl)). }
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [Hclosure_call | (closure & applied & applied_arity & _ & -> & _ & _)].
  - (* The callee is a closure. *)
    destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hclosure_call)
      as (id & slots & c & vs' & arity' & Hof & Hc & _ & Hargs & Hcode_call).
    destruct (called_code id c Hc) as (Hin_codes & Hcheck & entry & Hentry).
    destruct (check_code_parts cert codes id c entry Hcheck Hentry)
      as (Hlen_flags & Hlen_params & Hindirect & _ & _).
    pose proof (arguments_for_code_forall _ _ _ _ _ _ _
                  (good_fields cert codes rs Hgood_statics) Hgood_vs Hargs)
      as Hgood_vs'.
    pose proof (arguments_for_code_forall (live rs) _ _ _ _ _ _
                  (fun size tuple fields Hl Hf =>
                     proj1 (tuple_placed rs Hfresh Hdeep Hstatics tuple size
                              fields Hl Hf))
                  Hlive_vs Hargs)
      as Hlive_vs'.
    assert (Hfit : fit_pointwise rs (ce_params_local entry) vs').
    { destruct (unknown_call_fit cs l_c ls _ l' Hunknown)
        as [Hforever | Hlocal_cert].
      - assert (Hboth :
          Forall
            (fun v => live rs v ∧ value_all (heap_symbol rs) v = true) vs').
        { apply (arguments_for_code_forall _ _ _ _ _ _ _) with (3 := Hargs).
          - intros size tuple fields [Hl Hh] Hf.
            destruct (tuple_placed rs Hfresh Hdeep Hstatics tuple size fields
                        Hl Hf)
              as [Hlive_fields Hheap_fields].
            specialize (Hheap_fields Hh). apply Forall_forall. intros w Hin.
            split;
              [ exact (proj1 (Forall_forall _ _) Hlive_fields w Hin)
              | exact (proj1 (Forall_forall _ _) Hheap_fields w Hin) ].
          - apply Forall_forall. intros w Hin.
            split;
              [ exact (proj1 (Forall_forall _ _) Hlive_vs w Hin)
              | exact (proj1 (Forall_forall _ _) (Hheap_vs Hforever) w Hin) ]. }
        intros i v flag Hv _. left.
        exact (proj2 (proj1 (Forall_forall _ _) Hboth v (nth_error_In _ _ Hv))).
      - intros i v flag _ Hflag. right.
        exact (forallb_id_nth _ i flag (Hindirect Hlocal_cert) Hflag). }
    apply (call_unknown_closure a l_c ls (length args_arity) l' callee id slots
             c entry vs' arity' return_arity return_to exn_to shape s' Hcaller
             Hunknown Hret Hraise Hreturns Hgood_ret Hgood_exn Hof Hc Hentry
             Hgood_callee Hgood_vs' Hlive_vs' Hfit);
      [ | | exact Hcode_call ].
    + intros Hlen Hkinds Hper. unfold arguments_for_code in Hargs.
      destruct (code_is_tupled c) eqn:Htupled.
      * destruct (detuple_cases _ _ _ _ _ _ Hargs)
          as (tuple & rest & first_c & rest_arity & fields & -> & -> & _ & _
              & Htuple & -> & ->).
        pose proof (tuple_fields_length _ _ _ _ Htuple) as Hfields.
        rewrite skipn_app, skipn_all2, Hfields, Nat.sub_diag by lia.
        cbn [app skipn].
        inversion Hholds as [ | l_t v_t ls_rest vs_rest Ht Hrest]; subst.
        split.
        -- apply (later_call_code codes id c _ (lookup_in _ _ _ Hin_codes)).
           unfold full_arity. rewrite Htupled. cbn [length].
           destruct rest_arity as [ | component rest_arity];
             [ exfalso | cbn [length]; lia ].
           rewrite app_nil_r in Hlen, Hper.
           unfold arity_of_kinds in Hlen, Hper.
           rewrite length_map, repeat_length in Hlen.
           assert (Hcardinal :
             length
               (concat
                  (unarize_per_parameter
                     (firstn (length (code_params_arity c))
                        (map Singleton
                           (repeat Value
                              (cardinal_unarized (code_params_arity c)))))))
             = length (code_params_arity c)).
           { rewrite singletons_cardinal.
             - rewrite length_firstn, length_map, repeat_length. lia.
             - intros component Hin. apply In_firstn in Hin.
               apply in_map_iff in Hin as (kind & <- & _). exists kind.
               reflexivity. }
           rewrite Hper, <- unarize_concat in Hcardinal.
           unfold cardinal_unarized in Hlen. lia.
        -- exists ls_rest. split; [ exact Hrest | ].
           intros l0 Hin. right. exact Hin.
      * injection Hargs as <- <-. split.
        -- apply (later_call_code codes id c _ (lookup_in _ _ _ Hin_codes)).
           unfold full_arity. rewrite Htupled. exact Hlen.
        -- exists (skipn (cardinal_unarized (code_params_arity c)) ls).
           split; [ apply Forall2_skipn; exact Hholds | ].
           intros l0. apply In_skipn.
    + intros Hlen Htupled Hkinds. unfold arguments_for_code in Hargs.
      rewrite Htupled in Hargs. injection Hargs as <- <-.
      assert (Hwide : length args_arity < max_full_arity codes).
      { pose proof (full_arity_le_max codes id c (lookup_in _ _ _ Hin_codes))
          as Hmax.
        unfold full_arity in Hmax. rewrite Htupled in Hmax. lia. }
      split; [ exact Hwide | ].
      split; [ exact (Hkept_callee callee Hholds_callee) | ].
      split; [ exact Hkept_vs | ].
      intros Hstrict_cert.
      destruct (unknown_call_strict cs l_c ls _ l' Hstrict_cert Hunknown)
        as (_ & Hargs_forever & Hcallee_wide).
      split.
      * destruct (is_forever l_c) eqn:Hforever.
        -- destruct l_c; try discriminate. exact Hholds_callee.
        -- destruct (closure_place cert codes rs Hlocal callee id slots entry
                       Hof Hgood_callee Hentry)
             as [_ Hheap].
           destruct (ce_closure_local entry) eqn:Hclosure_local;
             [ exfalso | exact (Hheap eq_refl) ].
           pose proof (full_arity_le_max_local cert codes id c entry
                         (lookup_in _ _ _ Hin_codes) Hentry Hclosure_local)
             as Hmax.
           unfold full_arity in Hmax. rewrite Htupled in Hmax.
           specialize (Hcallee_wide eq_refl). lia.
      * apply Hheap_vs. apply Hargs_forever. unfold may_be_partial.
        apply orb_true_iff. left. apply Nat.ltb_lt. exact Hwide.
  - (* The callee is a partial application. *)
    destruct (shape_of_partial_callee _ _ _ _ _ _ _ _ _ Hshape)
      as (_ & _ & Hkinds_vs & Hclosure_call).
    destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hclosure_call)
      as (id & slots & c & vs' & arity' & Hof & Hc & _ & Hargs & Hcode_call).
    destruct (called_code id c Hc) as (Hin_codes & Hcheck & entry & Hentry).
    destruct (check_code_parts cert codes id c entry Hcheck Hentry)
      as (Hlen_flags & Hlen_params & Hindirect & _ & _).
    destruct (good_partial cert codes rs _ _ _ Hgood_callee)
      as [Hgood_closure Hgood_applied].
    assert (Hpap :
      pap_fits cert codes rs closure applied applied_arity = true).
    { destruct Hgood_callee as [_ H]. cbn [value_paps] in H.
      apply andb_true_iff in H as [H _]. apply andb_true_iff in H as [H _].
      exact H. }
    unfold pap_fits in Hpap. rewrite Hof, Hin_codes, Hentry in Hpap.
    apply andb_true_iff in Hpap as [Hpap Hkeeps].
    apply andb_true_iff in Hpap as [Hnot_tupled Hfewer].
    apply negb_true_iff in Hnot_tupled. apply Nat.ltb_lt in Hfewer.
    unfold arguments_for_code in Hargs. rewrite Hnot_tupled in Hargs.
    injection Hargs as <- <-.
    apply (live_partial rs) in Hlive_callee as [Hlive_closure Hlive_applied].
    apply holds_partial in Hholds_callee as [Hholds_closure Hholds_applied].
    assert (Hfit : fit_pointwise rs (ce_params_local entry) (applied ++ vs)).
    { intros i v flag Hv Hflag.
      destruct (Nat.lt_ge_cases i (length applied)) as [Hi | Hi].
      - rewrite nth_error_app1 in Hv by exact Hi.
        apply orb_true_iff in Hkeeps as [H | H];
          apply andb_true_iff in H as [_ H].
        + left. exact (forallb_in _ _ v H (nth_error_In _ _ Hv)).
        + exact (values_fit_nth rs _ applied i v flag H Hv Hflag).
      - rewrite nth_error_app2 in Hv by exact Hi.
        destruct (unknown_call_fit cs l_c ls _ l' Hunknown)
          as [Hforever | Hlocal_cert].
        + left.
          exact (proj1 (Forall_forall _ _) (Hheap_vs Hforever) v
                   (nth_error_In _ _ Hv)).
        + right.
          exact (forallb_id_nth _ i flag (Hindirect Hlocal_cert) Hflag). }
    apply (call_unknown_closure a l_c ls (length args_arity) l' closure id
             slots c entry (applied ++ vs) (applied_arity ++ args_arity)
             return_arity return_to exn_to shape s' Hcaller Hunknown Hret
             Hraise Hreturns Hgood_ret Hgood_exn Hof Hc Hentry Hgood_closure
             (proj2 (Forall_app _ _ _) (conj Hgood_applied Hgood_vs))
             (proj2 (Forall_app _ _ _) (conj Hlive_applied Hlive_vs)) Hfit);
      [ | | exact Hcode_call ].
    + intros Hlen Hkinds Hper. rewrite length_app in Hlen.
      pose proof (full_arity_le_max codes id c (lookup_in _ _ _ Hin_codes))
        as Hmax.
      unfold full_arity in Hmax. rewrite Hnot_tupled in Hmax.
      split.
      * destruct (Nat.lt_ge_cases 1 (length (code_params_arity c)))
          as [Hwide | Hnarrow].
        -- apply later_call_wide; lia.
        -- apply (later_call_code codes id c _ (lookup_in _ _ _ Hin_codes)).
           unfold full_arity. rewrite Hnot_tupled. lia.
      * assert (Happlied : length applied = length (unarize applied_arity)).
        { apply (f_equal (@length _)) in Hkinds, Hkinds_vs.
          rewrite length_map, unarize_app, !length_app in Hkinds.
          rewrite length_map in Hkinds_vs. lia. }
        assert (Hsize :
          length applied <= cardinal_unarized (code_params_arity c)).
        { unfold cardinal_unarized.
          rewrite unarize_concat, <- Hper, firstn_app, firstn_all2 by lia.
          unfold unarize_per_parameter. rewrite map_app, concat_app, length_app.
          change (map unarize_component applied_arity)
            with (unarize_per_parameter applied_arity).
          rewrite <- unarize_concat. lia. }
        rewrite skipn_app, skipn_all2 by exact Hsize. cbn [app].
        exists (skipn (cardinal_unarized (code_params_arity c) - length applied)
                  ls).
        split; [ apply Forall2_skipn; exact Hholds | ].
        intros l0. apply In_skipn.
    + intros Hlen Htupled Hkinds. rewrite length_app in Hlen.
      assert (Hwide : length args_arity < max_full_arity codes).
      { pose proof (full_arity_le_max codes id c (lookup_in _ _ _ Hin_codes))
          as Hmax.
        unfold full_arity in Hmax. rewrite Htupled in Hmax. lia. }
      split; [ exact Hwide | ].
      split; [ exact (Hkept_callee closure Hholds_closure) | ].
      split.
      * apply Forall_app. split; [ | exact Hkept_vs ].
        revert Hholds_applied. apply Forall_impl. exact Hkept_callee.
      * intros Hstrict_cert. rewrite Hstrict_cert in Hkeeps.
        cbn [andb] in Hkeeps. rewrite orb_false_r in Hkeeps.
        apply andb_true_iff in Hkeeps as [Hheap_closure Hheap_applied].
        split; [ exact Hheap_closure | ].
        apply Forall_app.
        split; [ exact (proj1 (forallb_Forall _ _) Hheap_applied) | ].
        destruct (unknown_call_strict cs l_c ls _ l' Hstrict_cert Hunknown)
          as (_ & Hargs_forever & _).
        apply Hheap_vs. apply Hargs_forever. unfold may_be_partial.
        apply orb_true_iff. left. apply Nat.ltb_lt. exact Hwide.
Qed.

(* ======================================================================== *)
(* A direct call                                                            *)
(* ======================================================================== *)

(* What the check says of a direct call of the code [id]:
   FlambdaRegionsCheckSpec.call_result at a call that is direct. *)
Definition direct_result (cs : check_state) (id : code_id) (callee : lifetime)
    (args : list lifetime) (count : nat) : option lifetime :=
  let* c := lookup id codes in
  let* entry := lookup id (cert_code cert) in
  let flags :=
    if code_is_tupled c
    then [forallb (fun flag => flag) (ce_params_local entry)]
    else ce_params_local entry
  in
  if args_fit (cert_indirect_local cert) flags args
  then
    if count <? full_arity c
    then
      let kept := shortest cs (callee :: args) in
      if cert_partial_local cert then Some kept
      else if is_forever kept then Some Forever
      else None
    else
      let first :=
        if ce_returns_local entry then newest_lifetime cs else Forever
      in
      if count =? full_arity c then Some first
      else
        unknown_call cert codes cs first (skipn (length flags) args)
          (count - full_arity c)
  else None.

Lemma args_fit_nth (indirect_local : bool) (flags : list bool) :
  ∀ (ls : list lifetime) (i : nat) (l : lifetime) (flag : bool),
    args_fit indirect_local flags ls = true ->
    nth_error ls i = Some l ->
    nth_error flags i = Some flag ->
    is_forever l = true ∨ flag = true.
Proof.
  induction flags as [ | flag0 flags IH]; intros [ | l0 ls] [ | i] l flag;
    cbn [args_fit nth_error]; try discriminate.
  - intros H [= <-] [= <-]. apply andb_true_iff in H as [H _].
    apply orb_true_iff in H. exact H.
  - intros H. apply andb_true_iff in H as [_ H]. exact (IH ls i l flag H).
Qed.

(* The parameters of tupled code that a call with more arguments
   reaches are at least the fields of the tuple. *)
Lemma tupled_over_params (params_arity rest_arity : arity) :
  unarize_per_parameter
    (firstn (length params_arity)
       (arity_of_kinds (repeat Value (cardinal_unarized params_arity))
        ++ rest_arity))
  = unarize_per_parameter params_arity ->
  cardinal_unarized params_arity <= length params_arity.
Proof.
  intros Hper.
  destruct (Nat.le_gt_cases (cardinal_unarized params_arity)
              (length params_arity))
    as [Hle | Hgt]; [ exact Hle | exfalso ].
  unfold arity_of_kinds in Hper.
  assert (Hcardinal :
    length
      (concat
         (unarize_per_parameter
            (firstn (length params_arity)
               (map Singleton
                  (repeat Value (cardinal_unarized params_arity))
                ++ rest_arity))))
    = length params_arity).
  { rewrite singletons_cardinal.
    - rewrite length_firstn, length_app, length_map, repeat_length. lia.
    - intros component Hin.
      rewrite firstn_app, length_map, repeat_length in Hin.
      replace (length params_arity - cardinal_unarized params_arity) with 0
        in Hin by lia.
      cbn [firstn] in Hin. rewrite app_nil_r in Hin.
      apply In_firstn in Hin. apply in_map_iff in Hin as (kind & <- & _).
      exists kind. reflexivity. }
  rewrite Hper, <- unarize_concat in Hcardinal.
  unfold cardinal_unarized in Hgt. lia.
Qed.

Lemma call_direct (a : caller) (id : code_id) (callee : value)
    (l_c : lifetime) (ls : list lifetime) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (l' : lifetime) (s' : state) :
  caller_ok rs a ->
  good cert codes rs callee ->
  lifetime_holds rs (cs_open (caller_check a)) (caller_marks a) None l_c
    callee ->
  Forall (good cert codes rs) vs ->
  Forall2 (caller_holds rs a) ls vs ->
  direct_result (caller_check a) id l_c ls (length args_arity) = Some l' ->
  ret_ok cert codes rs a l' return_to ->
  raise_ok cert codes rs a exn_to ->
  returns_ok a l' return_to ->
  cont_values_all (good cert codes rs) return_to ->
  cont_values_all (good cert codes rs) exn_to ->
  call_function callee (Direct id) vs args_arity return_arity return_to
    exn_to (live_statics rs) hp
  = Step s' ->
  entered cert codes rs s'.
Proof.
  intros Hcaller Hgood_callee Hholds_callee Hgood_vs Hholds Hdirect Hret
    Hraise Hreturns Hgood_ret Hgood_exn.
  unfold call_function.
  destruct (shape_of_call (live_statics rs) callee (Direct id) vs args_arity
              return_arity)
    as [shape | ] eqn:Hshape; [ | discriminate ].
  set (cs := caller_check a) in *.
  set (kept := shortest cs (l_c :: ls)).
  assert (Hfacts :
    Forall2
      (fun l v =>
         live rs v
         ∧ lifetime_holds rs (cs_open cs) (caller_marks a) None l v
         ∧ (is_forever l = true -> value_all (heap_symbol rs) v = true))
      ls vs).
  { revert Hholds. apply Forall2_impl. intros l v.
    exact (caller_holds_facts a l v Hcaller). }
  assert (Hlive_vs : Forall (live rs) vs).
  { apply (Forall2_right_in _ _ ls vs Hfacts). intros l v _ H.
    exact (proj1 H). }
  assert (Hkept_vs :
    Forall (lifetime_holds rs (cs_open cs) (caller_marks a) None kept) vs).
  { apply (Forall2_right_in _ _ ls vs Hfacts). intros l v Hin H.
    apply (holds_rank rs (caller_vars a) _ _ l kept v (caller_marked a Hcaller)
             (caller_placed rs a Hcaller));
      [ | exact (proj1 (proj2 H)) ].
    exact (rank_shortest_le cs (l_c :: ls) l (or_intror Hin)). }
  assert (Hkept_callee :
    lifetime_holds rs (cs_open cs) (caller_marks a) None kept callee).
  { apply (holds_rank rs (caller_vars a) _ _ l_c kept callee
             (caller_marked a Hcaller) (caller_placed rs a Hcaller));
      [ | exact Hholds_callee ].
    exact (rank_shortest_le cs (l_c :: ls) l_c (or_introl eq_refl)). }
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [Hclosure_call | (closure & applied & applied_arity & H & _)];
    [ | discriminate H ].
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hclosure_call)
    as (called & slots & c & vs' & arity' & Hof & Hc & Hnamed & Hargs
        & Hcode_call).
  rewrite (Hnamed id eq_refl) in Hof, Hc. clear Hnamed called.
  destruct (called_code id c Hc) as (Hin_codes & Hcheck & entry & Hentry).
  destruct (check_code_parts cert codes id c entry Hcheck Hentry)
    as (Hlen_flags & Hlen_params & _ & _ & Hreturns_local).
  destruct (closure_place cert codes rs Hlocal callee id slots entry Hof
              Hgood_callee Hentry)
    as [Hlive_callee _].
  unfold direct_result in Hdirect. rewrite Hin_codes, Hentry in Hdirect.
  cbv beta iota zeta in Hdirect.
  set (first :=
    if ce_returns_local entry then newest_lifetime cs else Forever) in *.
  set (flags :=
    if code_is_tupled c
    then [forallb (fun flag => flag) (ce_params_local entry)]
    else ce_params_local entry) in *.
  destruct (args_fit (cert_indirect_local cert) flags ls) eqn:Hargs_fit;
    [ | discriminate ].
  assert (Hfirst : first = Forever ∨ first = unknown_lifetime cert cs).
  { unfold first, unknown_lifetime.
    destruct (ce_returns_local entry); [ | left; reflexivity ].
    rewrite (Hreturns_local eq_refl). right. reflexivity. }
  assert (Hfirst_rank :
    rank (cs_open cs) (unknown_lifetime cert cs) <= rank (cs_open cs) first).
  { destruct Hfirst as [-> | ->]; [ | lia ].
    pose proof (rank_le (cs_open cs) (unknown_lifetime cert cs)).
    cbn [rank]. lia. }
  pose proof (arguments_for_code_forall _ _ _ _ _ _ _
                (good_fields cert codes rs Hgood_statics) Hgood_vs Hargs)
    as Hgood_vs'.
  pose proof (arguments_for_code_forall (live rs) _ _ _ _ _ _
                (fun size tuple fields Hl Hf =>
                   proj1 (tuple_placed rs Hfresh Hdeep Hstatics tuple size
                            fields Hl Hf))
                Hlive_vs Hargs)
    as Hlive_vs'.
  unfold arguments_for_code in Hargs.
  destruct (code_is_tupled c) eqn:Htupled.
  - (* Tupled code. *)
    destruct (detuple_cases _ _ _ _ _ _ Hargs)
      as (tuple & rest & first_c & rest_arity & fields & -> & -> & _ & _
          & Htuple & -> & ->).
    pose proof (tuple_fields_length _ _ _ _ Htuple) as Hfields.
    inversion Hholds as [ | l_t v_t ls_rest vs_rest Ht Hrest]; subst.
    inversion Hfacts as [ | l_t0 v_t0 ls_rest0 vs_rest0 Ht_facts _]; subst.
    destruct Ht_facts as (Hlive_tuple & _ & Hheap_tuple).
    unfold flags in Hargs_fit. cbn [args_fit] in Hargs_fit.
    apply andb_true_iff in Hargs_fit as [Htuple_fit _].
    unfold full_arity in Hdirect. rewrite Htupled in Hdirect.
    cbn [length] in Hdirect.
    assert (Hfit :
      fit_pointwise rs (ce_params_local entry) (fields ++ rest)).
    { intros i v flag Hv Hflag.
      assert (Hi : i < length fields).
      { rewrite Hfields, <- Hlen_params, <- Hlen_flags.
        apply nth_error_Some. rewrite Hflag. discriminate. }
      rewrite nth_error_app1 in Hv by exact Hi.
      apply orb_true_iff in Htuple_fit as [Hforever | Hall].
      - left.
        destruct (tuple_placed rs Hfresh Hdeep Hstatics tuple _ fields
                    Hlive_tuple Htuple)
          as [_ Hheap_fields].
        exact (proj1 (Forall_forall _ _) (Hheap_fields (Hheap_tuple Hforever)) v
                 (nth_error_In _ _ Hv)).
      - right. exact (forallb_id_nth _ i flag Hall Hflag). }
    apply (call_closure a callee id slots c entry (fields ++ rest) _
             return_arity return_to exn_to shape s' Hcaller Hof Hc Hentry
             Hgood_callee Hgood_vs' Hlive_vs' Hfit Hraise Hgood_ret Hgood_exn
             Hcode_call).
    + intros _.
      destruct (length rest_arity) as [ | more] eqn:Hmore;
        cbn [Nat.ltb Nat.leb Nat.eqb] in Hdirect.
      * injection Hdirect as <-. exact Hret.
      * apply (ret_longer cert codes rs a first l' return_to Hret).
        exact
          (Nat.le_trans _ _ _
             (unknown_call_le cert codes cs _ _ _ l' Hdirect) Hfirst_rank).
    + intros Hlen _ Hper.
      pose proof (tupled_over_params (code_params_arity c) rest_arity Hper)
        as Hsize.
      rewrite skipn_app, skipn_all2, Hfields, Nat.sub_diag by lia.
      cbn [app skipn].
      unfold arity_of_kinds in Hlen.
      rewrite length_app, length_map, repeat_length in Hlen.
      assert (Hcount' :
        1 <= length
               (partially_apply
                  (arity_of_kinds
                     (repeat Value (cardinal_unarized (code_params_arity c)))
                   ++ rest_arity)
                  (length (code_params_arity c)))).
      { unfold partially_apply, arity_of_kinds.
        rewrite length_skipn, length_app, length_map, repeat_length. lia. }
      assert (Hcount'' :
        length
          (partially_apply
             (arity_of_kinds
                (repeat Value (cardinal_unarized (code_params_arity c)))
              ++ rest_arity)
             (length (code_params_arity c)))
        <= length rest_arity).
      { unfold partially_apply, arity_of_kinds.
        rewrite length_skipn, length_app, length_map, repeat_length. lia. }
      destruct (length rest_arity) as [ | more] eqn:Hmore; [ lia | ].
      unfold flags in Hdirect.
      cbn [Nat.ltb Nat.leb Nat.eqb skipn length Nat.sub] in Hdirect.
      destruct (Nat.eq_dec
                  (length
                     (partially_apply
                        (arity_of_kinds
                           (repeat Value
                              (cardinal_unarized (code_params_arity c)))
                         ++ rest_arity)
                        (length (code_params_arity c))))
                  (S more))
        as [Hsame | Hfewer].
      * rewrite <- Hsame in Hdirect.
        exact
          (Ret_over cert codes rs a first l' ls_rest rest _ return_arity
             return_to exn_to Hrest Hdirect Hret Hraise).
      * assert (Hlater : later_call codes (S more) = true).
        { apply (later_call_code codes id c _ (lookup_in _ _ _ Hin_codes)).
          unfold full_arity. rewrite Htupled. lia. }
        destruct (unknown_call_later cert codes cs first first ls_rest ls_rest
                    (S more) _ l' Hdirect Hlater Hcount' (fun l0 H => H)
                    Hfirst)
          as (l2 & Hcall2 & Hrank).
        exact
          (Ret_over cert codes rs a first l2 ls_rest rest _ return_arity
             return_to exn_to Hrest Hcall2
             (ret_longer cert codes rs a l2 l' return_to Hret Hrank) Hraise).
    + intros _ H. congruence.
  - (* Code that is not tupled. *)
    injection Hargs as <- <-.
    unfold full_arity in Hdirect. rewrite Htupled in Hdirect.
    assert (Hfit : fit_pointwise rs (ce_params_local entry) vs).
    { intros i v flag Hv Hflag.
      destruct (nth_error_same_length vs ls
                  (eq_sym (Forall2_length Hfacts)) i v Hv)
        as (l & Hl).
      destruct (args_fit_nth _ _ ls i l flag Hargs_fit Hl Hflag)
        as [Hforever | ->]; [ left | right; reflexivity ].
      exact (proj2 (proj2 (Forall2_nth_error _ _ _ Hfacts i l v Hl Hv))
               Hforever). }
    apply (call_closure a callee id slots c entry vs args_arity return_arity
             return_to exn_to shape s' Hcaller Hof Hc Hentry Hgood_callee
             Hgood_vs Hlive_vs Hfit Hraise Hgood_ret Hgood_exn Hcode_call).
    + intros Hlen. rewrite Hlen, Nat.ltb_irrefl, Nat.eqb_refl in Hdirect.
      injection Hdirect as <-. exact Hret.
    + intros Hlen _ _.
      assert (H1 :
        (length args_arity <? length (code_params_arity c)) = false)
        by (apply Nat.ltb_ge; lia).
      assert (H2 :
        (length args_arity =? length (code_params_arity c)) = false)
        by (apply Nat.eqb_neq; lia).
      rewrite H1, H2 in Hdirect. unfold flags in Hdirect.
      unfold cardinal_unarized in Hlen_params.
      rewrite Hlen_flags, Hlen_params in Hdirect.
      rewrite <- (length_skipn (length (code_params_arity c)) args_arity)
        in Hdirect.
      exact
        (Ret_over cert codes rs a first l' _ _ _ return_arity return_to exn_to
           (Forall2_skipn _ _ _ Hholds _) Hdirect Hret Hraise).
    + intros Hlen _ Hkinds Hpo s'' Hcall.
      assert (H1 :
        (length args_arity <? length (code_params_arity c)) = true)
        by (apply Nat.ltb_lt; exact Hlen).
      rewrite H1 in Hdirect. fold kept in Hdirect.
      assert (Hlength : length vs <= length (ce_params_local entry)).
      { rewrite Hlen_flags, Hlen_params.
        rewrite <- (length_map kind_of_value vs), Hkinds.
        unfold partial_or_over_application in Hpo.
        apply andb_true_iff in Hpo as [Hagree _].
        apply (prefix_cardinal _ _ Hagree). lia. }
      assert (Hforever_kept :
        is_forever kept = true ->
        value_all (heap_symbol rs) callee = true
        ∧ Forall (fun v => value_all (heap_symbol rs) v = true) vs).
      { intros Hforever. apply (is_forever_rank (cs_open cs)) in Hforever.
        split.
        - pose proof (rank_shortest_le cs (l_c :: ls) l_c (or_introl eq_refl))
            as Hle.
          pose proof (rank_le (cs_open cs) l_c).
          assert (Hc_forever : is_forever l_c = true)
            by (apply (is_forever_rank (cs_open cs)); fold kept in Hle; lia).
          destruct l_c; try discriminate. exact Hholds_callee.
        - apply (Forall2_right_in _ _ ls vs Hfacts). intros l v Hin H.
          apply (proj2 (proj2 H)). apply (is_forever_rank (cs_open cs)).
          pose proof (rank_shortest_le cs (l_c :: ls) l (or_intror Hin)) as Hle.
          pose proof (rank_le (cs_open cs) l). fold kept in Hle. lia. }
      assert (Hpap_fits :
        pap_fits cert codes rs callee vs args_arity = true).
      { unfold pap_fits. rewrite Hof, Hin_codes, Hentry, Htupled, H1.
        cbn [negb andb].
        destruct (cert_partial_local cert) eqn:Hlocal_cert.
        - rewrite (pointwise_values_fit rs _ vs Hlength Hfit). apply orb_true_r.
        - destruct (is_forever kept) eqn:Hforever; [ | discriminate ].
          destruct (Hforever_kept eq_refl) as [Hheap_callee Hheap_vs].
          rewrite Hheap_callee, (proj2 (forallb_Forall _ _) Hheap_vs).
          reflexivity. }
      apply (Hreturns [Value_partial_application callee vs args_arity] s'');
        [ | | | exact Hcall ].
      * apply Forall_cons; [ | apply Forall_nil ].
        destruct Hgood_callee as [Hplain_callee Hfits_callee].
        split.
        -- cbn [value_plain]. rewrite Hplain_callee. cbn [andb].
           apply forallb_forall. intros v Hin.
           exact (proj1 (proj1 (Forall_forall _ _) Hgood_vs v Hin)).
        -- cbn [value_paps]. rewrite Hpap_fits, Hfits_callee. cbn [andb].
           apply forallb_forall. intros v Hin.
           exact (proj2 (proj1 (Forall_forall _ _) Hgood_vs v Hin)).
      * apply Forall_cons; [ | apply Forall_nil ].
        apply live_partial. split; assumption.
      * apply Forall_cons; [ | apply Forall_nil ].
        destruct (cert_partial_local cert) eqn:Hlocal_cert.
        -- injection Hdirect as <-. apply holds_partial.
           split; [ exact Hkept_callee | exact Hkept_vs ].
        -- destruct (is_forever kept) eqn:Hforever; [ | discriminate ].
           injection Hdirect as <-.
           destruct (Hforever_kept eq_refl) as [Hheap_callee Hheap_vs].
           cbn [lifetime_holds value_all]. rewrite Hheap_callee.
           exact (proj2 (forallb_Forall _ _) Hheap_vs).
Qed.

End Call.
