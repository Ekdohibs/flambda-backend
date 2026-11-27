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

(* That the rebuild of the reaper commutes with the erasure of regions
   on the terms of FlambdaReaperRegionsSpec.regions_plain, for any facts
   and solutions that agree on what the rebuild reads.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperRebuild FlambdaRegionsSpec
  FlambdaReaperRegionsSpec.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs FlambdaReaperFlowProofs.
Import ListNotations.

(* ======================================================================== *)
(* The erasure of terms with no region parameter                            *)
(* ======================================================================== *)

Lemma drop_regions_plain {A : Type} (places : list bool) (l : list A) :
  forallb negb places = true -> drop_regions places l = l.
Proof.
  revert l. induction places as [ | p places IH]; intros l H;
    [ destruct l; reflexivity | ].
  cbn in H. apply andb_true_iff in H as [Hp H].
  destruct p; [ discriminate Hp | ].
  destruct l as [ | a l]; [ reflexivity | ]. cbn. rewrite (IH l H).
  reflexivity.
Qed.

(* Two tables of continuations that erase every [Apply_cont] alike. *)
Definition ceq (c c' : region_conts) : Prop :=
  ∀ ac : apply_cont, erase_apply_cont c ac = erase_apply_cont c' ac.

Lemma ceq_cons (k : continuation) (p : list bool) (c c' : region_conts) :
  ceq c c' -> ceq ((k, p) :: c) ((k, p) :: c').
Proof.
  intros H ac. specialize (H ac). unfold erase_apply_cont in *. cbn.
  destruct (Pos.eqb (ac_continuation ac) k); [ reflexivity | exact H ].
Qed.

Lemma ceq_app (l c c' : region_conts) : ceq c c' -> ceq (l ++ c) (l ++ c').
Proof.
  induction l as [ | [k p] l IH]; intros H; [ exact H | ].
  apply ceq_cons, IH, H.
Qed.

(* A table none of whose continuations has a region parameter. *)
Definition conts_plain (c : region_conts) : Prop :=
  ∀ (k : continuation) (p : list bool),
    lookup k c = Some p -> forallb negb p = true.

Lemma ceq_plain_nil (c : region_conts) : conts_plain c -> ceq c [].
Proof.
  intros H ac. unfold erase_apply_cont. cbn.
  destruct (lookup (ac_continuation ac) c) as [p | ] eqn:Hl;
    [ | reflexivity ].
  rewrite (drop_regions_plain p _ (H _ _ Hl)).
  destruct ac; reflexivity.
Qed.

Lemma erase_apply_cont_nil (ac : apply_cont) : erase_apply_cont [] ac = ac.
Proof. reflexivity. Qed.

Lemma erase_ceq :
  (∀ e : expr, ∀ c c', ceq c c' -> erase c e = erase c' e)
  ∧ (∀ h : continuation_handler,
       ∀ c c', ceq c c' -> erase_handler c h = erase_handler c' h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c c' H.
    destruct d as [s | p args | | | | | ]; cbn [erase];
      rewrite (IH c c' H); reflexivity.
  - intros bindings body _ IH c c' H. cbn [erase].
    rewrite (IH c c' H). reflexivity.
  - intros k h body IHh IH c c' H. cbn [erase].
    rewrite (IHh c c' H), (IH _ _ (ceq_cons _ _ _ _ H)). reflexivity.
  - intros handlers body IHh IH c c' H. cbn [erase].
    set (l := map (fun '(k, h) => (k, region_params h)) handlers).
    assert (H' : ceq (l ++ c) (l ++ c')) by exact (ceq_app l c c' H).
    rewrite (IH _ _ H'). f_equal.
    apply map_ext_in. intros [k h] Hin. rewrite (IHh k h Hin _ _ H').
    reflexivity.
  - intros ap c c' H. reflexivity.
  - intros ac c c' H. cbn [erase]. rewrite (H ac). reflexivity.
  - intros scrutinee arms c c' H. cbn [erase]. f_equal.
    apply map_ext. intros [i ac]. rewrite (H ac). reflexivity.
  - reflexivity.
  - intros params metadata handler IH c c' H. cbn [erase_handler].
    rewrite (IH c c' H). reflexivity.
Qed.

Lemma erase_plain (c : region_conts) (e : expr) :
  conts_plain c -> erase c e = erase [] e.
Proof. intros H. exact (proj1 erase_ceq e c [] (ceq_plain_nil c H)). Qed.

Lemma erase_handler_plain (c : region_conts) (h : continuation_handler) :
  conts_plain c -> erase_handler c h = erase_handler [] h.
Proof. intros H. exact (proj2 erase_ceq h c [] (ceq_plain_nil c H)). Qed.

Lemma conts_plain_nil : conts_plain [].
Proof. intros k p H. discriminate H. Qed.

Lemma conts_plain_cons (k : continuation) (p : list bool) (c : region_conts) :
  forallb negb p = true -> conts_plain c -> conts_plain ((k, p) :: c).
Proof.
  intros Hp Hc k' p' Hl. cbn in Hl.
  destruct (Pos.eqb k' k); [ injection Hl as <-; exact Hp | ].
  exact (Hc _ _ Hl).
Qed.

Lemma conts_plain_app (l c : region_conts) :
  (∀ k p, (k, p) ∈ l -> forallb negb p = true) -> conts_plain c ->
  conts_plain (l ++ c).
Proof.
  induction l as [ | [k p] l IH]; intros Hl Hc; [ exact Hc | ].
  cbn. apply conts_plain_cons; [ apply (Hl k p); left; reflexivity | ].
  apply IH; [ | exact Hc ]. intros k' p' Hin. apply (Hl k' p'). right.
  exact Hin.
Qed.

(* The places of the regions of kinds none of which is [Region]. *)
Lemma region_places_plain (kinds : list flambda_kind) :
  forallb kind_not_region kinds = true ->
  forallb negb (map (fun kind => kind_eqb kind Region) kinds) = true.
Proof.
  induction kinds as [ | kind kinds IH]; intros H; [ reflexivity | ].
  cbn [forallb map] in H |- *. apply andb_true_iff in H as [Hk H].
  unfold kind_not_region in Hk. rewrite Hk, (IH H). reflexivity.
Qed.

(* A handler none of whose parameters is a region is erased in its body
   alone. *)
Lemma erase_handler_no_region (c : region_conts) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  forallb kind_not_region (hm_params_kinds metadata) = true ->
  erase_handler c (Handler params metadata handler)
  = Handler params metadata (erase c handler).
Proof.
  intros H. cbn [erase_handler].
  pose proof (region_places_plain _ H) as Hp.
  rewrite !(drop_regions_plain _ _ Hp).
  destruct metadata; reflexivity.
Qed.

(* The erasure changes no continuation. *)
Lemma apply_cont_continuations_erase (c : region_conts) (ac : apply_cont) :
  apply_cont_continuations (erase_apply_cont c ac)
  = apply_cont_continuations ac.
Proof.
  unfold erase_apply_cont. destruct (lookup (ac_continuation ac) c);
    reflexivity.
Qed.

Lemma free_continuations_erase :
  (∀ e : expr, ∀ c,
     free_continuations_of_expr (erase c e) = free_continuations_of_expr e)
  ∧ (∀ h : continuation_handler, ∀ c,
       free_continuations_of_handler (erase_handler c h)
       = free_continuations_of_handler h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c.
    destruct d as [s | p args | | | | | ]; cbn [erase];
      try (cbn [free_continuations_of_expr]; apply IH).
    destruct p as [ | | | | r]; try (cbn [free_continuations_of_expr];
                                     apply IH).
    destruct (erase_region_primitive x r args);
      cbn [free_continuations_of_expr]; apply IH.
  - intros bindings body Hcodes IH c. cbn [erase free_continuations_of_expr].
    rewrite IH. f_equal. rewrite flat_map_concat_map, map_map.
    rewrite flat_map_concat_map. f_equal. apply map_ext_in.
    intros b Hin. destruct b as [id cd | id | s const]; [ | reflexivity
                                                          | reflexivity ].
    destruct cd as [params mc ret exn md body'].
    cbn [erase_binding erase_code free_continuations_of_binding
           free_continuations_of_code].
    rewrite (Hcodes id _ Hin). reflexivity.
  - intros k h body IHh IH c. cbn [erase free_continuations_of_expr].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH c. cbn [erase free_continuations_of_expr].
    rewrite IH, map_map. f_equal.
    + apply map_ext. intros [k h]. reflexivity.
    + f_equal. rewrite flat_map_concat_map, map_map, flat_map_concat_map.
      f_equal. apply map_ext_in. intros [k h] Hin. apply (IHh k h Hin).
  - intros ap c. reflexivity.
  - intros ac c. cbn [erase free_continuations_of_expr].
    apply apply_cont_continuations_erase.
  - intros scrutinee arms c. cbn [erase free_continuations_of_expr].
    rewrite flat_map_concat_map, map_map, flat_map_concat_map. f_equal.
    apply map_ext. intros [i ac]. apply apply_cont_continuations_erase.
  - reflexivity.
  - intros params metadata handler IH c. cbn [erase_handler].
    cbn [free_continuations_of_handler]. apply IH.
Qed.

(* ======================================================================== *)
(* The environments of the two rebuilds                                     *)
(* ======================================================================== *)

(* The environment of the rebuild of the erased term ([en']) is that of
   the term ([en]) without the variables of the regions: the same
   continuations and projections, and the same kind for every variable
   that is not of the kind [Region] in [en]. *)
Definition env_rel (en en' : rebuild_env) : Prop :=
  env_conts en' = env_conts en
  ∧ env_projections en' = env_projections en
  ∧ ∀ x : variable,
      region_variable (env_kinds en) x = false ->
      lookup x (env_kinds en') = lookup x (env_kinds en).

Lemma env_rel_refl (en : rebuild_env) : env_rel en en.
Proof.
  split; [ reflexivity | split; [ reflexivity | ] ]. intros; reflexivity.
Qed.

Lemma region_variable_app (l k : finite_map flambda_kind) (x : variable) :
  region_variable (l ++ k) x
  = match lookup x l with
    | Some kind => kind_eqb kind Region
    | None => region_variable k x
    end.
Proof.
  unfold region_variable. rewrite lookup_app.
  destruct (lookup x l); reflexivity.
Qed.

Lemma kinds_rel_app (l k k' : finite_map flambda_kind) :
  (∀ x, region_variable k x = false -> lookup x k' = lookup x k) ->
  ∀ x, region_variable (l ++ k) x = false ->
    lookup x (l ++ k') = lookup x (l ++ k).
Proof.
  intros H x Hx. rewrite region_variable_app in Hx. rewrite !lookup_app.
  destruct (lookup x l); [ reflexivity | exact (H x Hx) ].
Qed.

Lemma env_rel_bind_then (en en' : rebuild_env)
    (bound : list (variable * flambda_kind)) (ps : list variable) :
  env_rel en en' ->
  env_rel (bind_variables_then en bound ps) (bind_variables_then en' bound ps).
Proof.
  intros (Hc & _ & Hk). split; [ exact Hc | split; [ reflexivity | ] ].
  exact (kinds_rel_app _ _ _ Hk).
Qed.

Lemma env_rel_bind (en en' : rebuild_env)
    (bound : list (variable * flambda_kind)) :
  env_rel en en' ->
  env_rel (bind_variables en bound) (bind_variables en' bound).
Proof. apply env_rel_bind_then. Qed.

Lemma env_rel_bind_continuations (en en' : rebuild_env)
    (bound : list (continuation * list bool)) :
  env_rel en en' ->
  env_rel (bind_continuations en bound) (bind_continuations en' bound).
Proof.
  intros (Hc & _ & Hk). split; [ cbn; rewrite Hc; reflexivity | ].
  split; [ reflexivity | exact Hk ].
Qed.

(* The variable of a region is in [en] alone. *)
Lemma env_rel_bind_region (en en' : rebuild_env) (r : variable) :
  env_rel en en' -> env_projections en = [] ->
  env_rel (bind_variables en [(r, Region)]) en'.
Proof.
  intros (Hc & Hp & Hk) Hnil. split; [ exact Hc | ].
  split; [ rewrite Hp, Hnil; reflexivity | ].
  intros x Hx. unfold region_variable in Hx.
  cbn [bind_variables bind_variables_then env_kinds rev app lookup]
    in Hx |- *.
  destruct (Pos.eqb x r);
    [ rewrite FlambdaPrimitiveProofs.kind_eqb_refl in Hx; discriminate Hx
    | exact (Hk x Hx) ].
Qed.

(* A variable bound twice in [en], once in [en']. *)
Lemma env_rel_bind_twice (en en' : rebuild_env) (x : variable)
    (kind : flambda_kind) :
  env_rel en en' ->
  env_rel (bind_variables (bind_variables en [(x, kind)]) [(x, kind)])
    (bind_variables en' [(x, kind)]).
Proof.
  intros (Hc & _ & Hk). split; [ exact Hc | split; [ reflexivity | ] ].
  intros y Hy. unfold region_variable in Hy.
  cbn [bind_variables bind_variables_then env_kinds rev app lookup]
    in Hy |- *.
  destruct (Pos.eqb y x); [ reflexivity | exact (Hk y Hy) ].
Qed.

Lemma kind_of_variable_rel (en en' : rebuild_env) (x : variable) :
  env_rel en en' -> region_variable (env_kinds en) x = false ->
  kind_of_variable en' x = kind_of_variable en x.
Proof.
  intros (_ & _ & Hk) Hx. unfold kind_of_variable. rewrite (Hk x Hx).
  reflexivity.
Qed.

Lemma existsb_ext_in {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a, a ∈ l -> f a = g a) -> existsb f l = existsb g l.
Proof.
  induction l as [ | a l IH]; intros H; [ reflexivity | ].
  cbn. rewrite (H a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros a' Ha'. apply H. right. exact Ha'.
Qed.

Lemma forallb_ext_in {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a, a ∈ l -> f a = g a) -> forallb f l = forallb g l.
Proof.
  induction l as [ | a l IH]; intros H; [ reflexivity | ].
  cbn. rewrite (H a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros a' Ha'. apply H. right. exact Ha'.
Qed.

Lemma flat_map_ext_in {A B : Type} (f g : A -> list B) (l : list A) :
  (∀ a, a ∈ l -> f a = g a) -> flat_map f l = flat_map g l.
Proof.
  induction l as [ | a l IH]; intros H; [ reflexivity | ].
  cbn. rewrite (H a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros a' Ha'. apply H. right. exact Ha'.
Qed.

(* An arm of the [Switch] that [Keyed.compact] keeps is an arm. *)
Lemma in_compact_arms (arm : Z * apply_cont) (arms : list (Z * apply_cont)) :
  arm ∈ Keyed.compact Z.eqb arms -> arm ∈ arms.
Proof.
  destruct arm as [i ac]. intros Hin.
  apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hin.
  exact (keyed_lookup_in Z.eqb z_eqb_decides_equality _ _ _ Hin).
Qed.

(* ======================================================================== *)
(* The parts of the two rebuilds                                            *)
(* ======================================================================== *)

Section Commute.

(* The facts and the solution of the term ([F], [sol]) and of the erased
   term ([F'], [sol']): the facts that the rebuild reads are the same,
   and so is the solution but for the usage of the constants. *)
Variables F F' : facts.
Variables sol sol' : solution.
Variable b : bool.
Variable ucids : list code_id.
Variable ff : positive.

Hypothesis HC : f_constructor F' = f_constructor F.
Hypothesis HA : f_accessor F' = f_accessor F.
Hypothesis HP : f_parameter F' = f_parameter F.
Hypothesis Hpac :
  ∀ (x : node) (i : nat) (y : node),
    (x, i, y) ∈ f_parameter F -> y <> Node_all_constants.
Hypothesis Hu :
  ∀ n : node,
    n <> Node_all_constants -> s_has_usage sol' n = s_has_usage sol n.
Hypothesis Hsrc : s_has_source sol' = s_has_source sol.
Hypothesis Has : s_any_source sol' = s_any_source sol.
Hypothesis Hss : s_sources sol' = s_sources sol.
Hypothesis Hfu : s_field_used sol' = s_field_used sol.

Lemma var_used_eq (x : variable) (kind : flambda_kind) :
  is_var_used sol' x kind = is_var_used sol x kind.
Proof. destruct kind; cbn; try reflexivity; apply Hu; discriminate. Qed.

Lemma dead_var_eq (x : variable) (kind : flambda_kind) :
  is_dead_var sol' x kind = is_dead_var sol x kind.
Proof. unfold is_dead_var. rewrite Hsrc. reflexivity. Qed.

Lemma symbol_used_eq (s : symbol) :
  is_symbol_used sol' s = is_symbol_used sol s.
Proof. apply Hu. discriminate. Qed.

Lemma code_id_used_eq (id : code_id) :
  is_code_id_used sol' id = is_code_id_used sol id.
Proof. apply Hu. discriminate. Qed.

Section Env.

Variables en en' : rebuild_env.
Hypothesis Hen : env_rel en en'.

Lemma kind_eq (x : variable) :
  simple_not_region (env_kinds en) (Var x) = true ->
  kind_of_variable en' x = kind_of_variable en x.
Proof.
  intros H. apply (kind_of_variable_rel _ _ _ Hen).
  apply negb_true_iff in H. exact H.
Qed.

Lemma rewrite_simple_eq (s : simple) :
  simple_not_region (env_kinds en) s = true ->
  rewrite_simple sol' en' s = rewrite_simple sol en s.
Proof.
  intros H. destruct s as [x | sym | c]; cbn [rewrite_simple].
  - rewrite (kind_eq x H), var_used_eq. reflexivity.
  - rewrite symbol_used_eq. reflexivity.
  - reflexivity.
Qed.

Lemma map_rewrite_eq (l : list simple) :
  forallb (simple_not_region (env_kinds en)) l = true ->
  map (rewrite_simple sol' en') l = map (rewrite_simple sol en) l.
Proof.
  intros H. apply map_ext_in. intros s Hs.
  apply rewrite_simple_eq. rewrite forallb_forall in H. exact (H s Hs).
Qed.

Lemma poison_argument_eq (s : simple) :
  simple_not_region (env_kinds en) s = true ->
  poison_argument en' s = poison_argument en s.
Proof.
  intros H. destruct s as [x | sym | c]; cbn [poison_argument];
    [ rewrite (kind_eq x H) | | ]; reflexivity.
Qed.

Lemma rewrite_or_variable_eq (s : simple) :
  simple_not_region (env_kinds en) s = true ->
  rewrite_or_variable sol' en' s = rewrite_or_variable sol en s.
Proof.
  intros H. destruct s as [x | sym | c]; cbn [rewrite_or_variable];
    [ rewrite (kind_eq x H), var_used_eq | | ]; reflexivity.
Qed.

Lemma rebuild_fields_eq (base : node) (fields : list simple) :
  forallb (simple_not_region (env_kinds en)) fields = true ->
  ∀ i : nat,
    rebuild_fields sol' en' base i fields = rebuild_fields sol en base i fields.
Proof.
  induction fields as [ | s fields IH]; intros H i; [ reflexivity | ].
  cbn in H. apply andb_true_iff in H as [Hs H]. cbn [rebuild_fields].
  unfold field_used. rewrite Hfu, (rewrite_simple_eq s Hs), (IH H).
  reflexivity.
Qed.

Lemma rebuild_value_slots_eq (bound : list node)
    (value_slots : list (value_slot * simple)) :
  forallb (simple_not_region (env_kinds en)) (map snd value_slots) = true ->
  rebuild_value_slots sol' en' bound value_slots
  = rebuild_value_slots sol en bound value_slots.
Proof.
  intros H. unfold rebuild_value_slots, field_used. rewrite Hfu.
  apply flat_map_ext_in. intros [w s] Hin.
  rewrite forallb_forall in H.
  rewrite (rewrite_simple_eq s (H s (in_map snd _ _ Hin))). reflexivity.
Qed.

Lemma rebuild_set_eq (bound : list node) (d : named) :
  forallb (simple_not_region (env_kinds en)) (simples_of_named d) = true ->
  rebuild_set sol' en' bound d = rebuild_set sol en bound d.
Proof.
  intros H. destruct d; cbn [rebuild_set simples_of_named] in *;
    try reflexivity; rewrite (rebuild_value_slots_eq _ _ H); reflexivity.
Qed.

Lemma rebuild_named_eq (x : variable) (d : named) :
  forallb (simple_not_region (env_kinds en)) (simples_of_named d) = true ->
  rebuild_named sol' en' x d = rebuild_named sol en x d.
Proof.
  intros H. destruct d as [s | p args | | | | | ];
    cbn [rebuild_named simples_of_named] in *.
  - cbn in H. apply andb_true_iff in H as [H _].
    rewrite (rewrite_simple_eq s H). reflexivity.
  - destruct p as [u | bp | vp | hp | rp];
      try (rewrite (map_rewrite_eq args H); reflexivity).
    + destruct u; try (rewrite (map_rewrite_eq args H); reflexivity).
      unfold field_used. rewrite Hfu, (map_rewrite_eq args H). reflexivity.
    + destruct vp; try (rewrite (map_rewrite_eq args H); reflexivity).
      rewrite (rebuild_fields_eq _ _ H). reflexivity.
    + destruct hp; try (rewrite (map_rewrite_eq args H); reflexivity).
      rewrite (rebuild_fields_eq _ _ H). reflexivity.
  - reflexivity.
  - reflexivity.
  - cbn in H. apply andb_true_iff in H as [H _].
    rewrite (rewrite_simple_eq _ H). reflexivity.
  - cbn in H. apply andb_true_iff in H as [H _].
    rewrite (rewrite_simple_eq _ H). reflexivity.
  - cbn in H. apply andb_true_iff in H as [H _].
    rewrite (rewrite_simple_eq _ H). reflexivity.
Qed.

Lemma rebuild_static_const_eq (s : symbol) (const : static_const) :
  forallb (simple_not_region (env_kinds en)) (simples_of_static_const const)
  = true ->
  rebuild_static_const sol' en' s const = rebuild_static_const sol en s const.
Proof.
  intros H. destruct const; cbn [rebuild_static_const simples_of_static_const]
    in *.
  - rewrite (rebuild_value_slots_eq _ _ H). reflexivity.
  - rewrite (rebuild_value_slots_eq _ _ H). reflexivity.
  - rewrite (rebuild_fields_eq _ _ H). reflexivity.
  - cbn in H. apply andb_true_iff in H as [H _].
    rewrite (rewrite_or_variable_eq _ H). reflexivity.
  - reflexivity.
  - rewrite (map_rewrite_eq _ H). reflexivity.
Qed.

Lemma static_const_is_used_eq (s : symbol) (const : static_const) :
  static_const_is_used sol' s const = static_const_is_used sol s const.
Proof.
  destruct const; cbn [static_const_is_used]; try apply symbol_used_eq.
  apply existsb_ext_in. intros [f sym] _. apply symbol_used_eq.
Qed.

Lemma rewrite_apply_cont_eq (ac : apply_cont) :
  forallb (simple_not_region (env_kinds en)) (ac_args ac) = true ->
  rewrite_apply_cont sol' en' ac = rewrite_apply_cont sol en ac.
Proof.
  intros H. destruct Hen as (Hc & _ & _). unfold rewrite_apply_cont.
  unfold simple_has_no_source. rewrite Hsrc, Hc, (map_rewrite_eq _ H).
  reflexivity.
Qed.

Lemma rebuild_apply_cont_eq (ac : apply_cont) :
  forallb (simple_not_region (env_kinds en)) (ac_args ac) = true ->
  rebuild_apply_cont sol' en' ac = rebuild_apply_cont sol en ac.
Proof.
  intros H. unfold rebuild_apply_cont. rewrite (rewrite_apply_cont_eq ac H).
  reflexivity.
Qed.

Lemma rebuild_switch_eq (scrutinee : simple) (arms : list (Z * apply_cont)) :
  simple_not_region (env_kinds en) scrutinee = true ->
  forallb (fun '(_, ac) => forallb (simple_not_region (env_kinds en))
                             (ac_args ac)) arms = true ->
  rebuild_switch sol' en' scrutinee arms = rebuild_switch sol en scrutinee arms.
Proof.
  intros Hs Harms. unfold rebuild_switch.
  rewrite (rewrite_simple_eq _ Hs).
  rewrite (flat_map_ext_in
             (fun '(d, ac) =>
                match rewrite_apply_cont sol' en' ac with
                | Some ac0 => [(d, ac0)]
                | None => []
                end)
             (fun '(d, ac) =>
                match rewrite_apply_cont sol en ac with
                | Some ac0 => [(d, ac0)]
                | None => []
                end)); [ reflexivity | ].
  intros [d ac] Hin. apply in_compact_arms in Hin.
  rewrite forallb_forall in Harms.
  rewrite (rewrite_apply_cont_eq ac (Harms _ Hin)). reflexivity.
Qed.

Lemma mask_arguments_eq (mask : list bool) (args : list simple) :
  forallb (simple_not_region (env_kinds en)) args = true ->
  mask_arguments en' mask args = mask_arguments en mask args.
Proof.
  revert mask. induction args as [ | s args IH]; intros mask H;
    [ reflexivity | ].
  cbn in H. apply andb_true_iff in H as [Hs H].
  destruct mask as [ | keep mask]; cbn [mask_arguments];
    rewrite (IH _ H); [ reflexivity | ].
  rewrite (poison_argument_eq s Hs). reflexivity.
Qed.

Lemma simple_is_dead_eq (s : simple) :
  simple_not_region (env_kinds en) s = true ->
  simple_is_dead sol' en' s = simple_is_dead sol en s.
Proof.
  intros H. destruct s as [x | sym | c]; cbn [simple_is_dead].
  - rewrite (kind_eq x H), dead_var_eq. reflexivity.
  - rewrite Hsrc. reflexivity.
  - reflexivity.
Qed.

End Env.

Lemma params_to_keep_eq (is_exn : bool)
    (params : list (variable * flambda_kind)) :
  params_to_keep sol' is_exn params = params_to_keep sol is_exn params.
Proof.
  unfold params_to_keep. apply map_ext. intros [x kind].
  rewrite var_used_eq. reflexivity.
Qed.

Lemma handler_mask_eq (is_exn : bool) (h : continuation_handler) :
  handler_mask sol' is_exn h = handler_mask sol is_exn h.
Proof. destruct h. apply params_to_keep_eq. Qed.

(* The queries about calls. *)
Lemma get_direct_sources_eq (x : node) :
  get_direct_sources sol' x = get_direct_sources sol x.
Proof. unfold get_direct_sources. rewrite Has, Hss, Hsrc. reflexivity. Qed.

Lemma get_field_sources_eq (in_ : node -> bool) (f : field) :
  get_field_sources F' sol' in_ f = get_field_sources F sol in_ f.
Proof. unfold get_field_sources. rewrite HC, Has, Hss, Hsrc. reflexivity. Qed.

Lemma cofield_has_use_eq (in_ : node -> bool) (i : nat) :
  cofield_has_use F' sol' in_ i = cofield_has_use F sol in_ i.
Proof.
  unfold cofield_has_use. rewrite HP. apply existsb_ext_in.
  intros [[x j] y] Hin. rewrite (Hu y (Hpac x j y Hin)). reflexivity.
Qed.

Lemma arguments_used_by_call_eq (ep : closure_entry_point)
    (num_args : list nat) :
  ∀ src : allocation_sources,
    arguments_used_by_call F' sol' ep src num_args
    = arguments_used_by_call F sol ep src num_args.
Proof.
  induction num_args as [ | n rest IH]; intros src; [ reflexivity | ].
  cbn [arguments_used_by_call]. destruct src as [ | cs]; [ reflexivity | ].
  rewrite get_field_sources_eq.
  destruct (get_field_sources F sol cs (Field_call_witness ep)) as [ | ws];
    [ reflexivity | ].
  f_equal.
  - apply map_ext. intros i. apply cofield_has_use_eq.
  - destruct rest as [ | r rs]; [ reflexivity | ].
    rewrite get_field_sources_eq. apply IH.
Qed.

Lemma code_id_actually_directly_called_eq (closure : node) :
  code_id_actually_directly_called F' sol' closure
  = code_id_actually_directly_called F sol closure.
Proof.
  unfold code_id_actually_directly_called. rewrite HA, HC, Has, Hss, Hsrc.
  reflexivity.
Qed.

Lemma rewrite_call_kind_eq (ap : apply_expr) :
  rewrite_call_kind F' sol' b ucids ap = rewrite_call_kind F sol b ucids ap.
Proof.
  unfold rewrite_call_kind, called_call_kind.
  rewrite !code_id_actually_directly_called_eq. reflexivity.
Qed.

Section Env2.

Variables en en' : rebuild_env.
Hypothesis Hen : env_rel en en'.

Lemma rebuild_arguments_eq (ap : apply_expr) :
  forallb (simple_not_region (env_kinds en)) (ap_args ap) = true ->
  rebuild_arguments F' sol' en' ap = rebuild_arguments F sol en ap.
Proof.
  intros H. unfold rebuild_arguments.
  destruct (ap_callee ap); [ | | rewrite (map_rewrite_eq _ _ Hen _ H);
                                reflexivity ];
    rewrite (mask_arguments_eq _ _ Hen _ _ H);
    unfold arguments_used_by_known_arity_call,
      arguments_used_by_unknown_arity_call;
    rewrite !get_direct_sources_eq, !arguments_used_by_call_eq;
    reflexivity.
Qed.

Lemma rebuild_exn_extra_args_eq (ap : apply_expr) :
  forallb (simple_not_region (env_kinds en)) (exn_extra_simples ap) = true ->
  rebuild_exn_extra_args sol' en' ap = rebuild_exn_extra_args sol en ap.
Proof.
  intros H. destruct Hen as (Hc & _ & _). unfold rebuild_exn_extra_args.
  rewrite Hc. unfold exn_extra_simples in H.
  rewrite (map_ext_in
             (fun '(s, kind) => (rewrite_simple sol' en' s, kind))
             (fun '(s, kind) => (rewrite_simple sol en s, kind))).
  - reflexivity.
  - intros [s kind] Hin. rewrite forallb_forall in H.
    rewrite (rewrite_simple_eq _ _ Hen s (H s (in_map fst _ _ Hin))).
    reflexivity.
Qed.

Lemma make_apply_wrapper_eq (ap : apply_expr) :
  make_apply_wrapper ff None en' ap = make_apply_wrapper ff None en ap.
Proof.
  destruct Hen as (Hc & _ & _). unfold make_apply_wrapper. rewrite Hc.
  reflexivity.
Qed.

Lemma rebuild_apply_eq (ap : apply_expr) :
  forallb (simple_not_region (env_kinds en))
    (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  rebuild_apply F' sol' b ucids ff None en' ap
  = rebuild_apply F sol b ucids ff None en ap.
Proof.
  intros H. cbn [forallb] in H. apply andb_true_iff in H as [Hc H].
  rewrite forallb_app in H. apply andb_true_iff in H as [Hargs Hextra].
  unfold rebuild_apply.
  rewrite (simple_is_dead_eq _ _ Hen _ Hc).
  rewrite (existsb_ext_in (simple_is_dead sol' en') (simple_is_dead sol en)).
  - rewrite rewrite_call_kind_eq, (rebuild_arguments_eq _ Hargs),
      (rebuild_exn_extra_args_eq _ Hextra), make_apply_wrapper_eq.
    reflexivity.
  - intros s Hs. rewrite forallb_forall in Hargs.
    exact (simple_is_dead_eq _ _ Hen s (Hargs s Hs)).
Qed.

End Env2.

(* ======================================================================== *)
(* Views of the terms                                                       *)
(* ======================================================================== *)

(* The primitive of regions that a defining expression is, if it is
   one. *)
Definition region_prim_of (d : named)
    : option (region_primitive * list simple) :=
  match d with
  | Prim (Region_primitive p) args => Some (p, args)
  | _ => None
  end.

Lemma region_prim_of_some (d : named) (p : region_primitive)
    (args : list simple) :
  region_prim_of d = Some (p, args) -> d = Prim (Region_primitive p) args.
Proof.
  destruct d as [ | [ | | | | ] | | | | | ]; cbn; intros H;
    try discriminate H.
  injection H as -> ->. reflexivity.
Qed.

Lemma erase_let_plain (c : region_conts) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  region_prim_of d = None ->
  erase c (Let x kind d body) = Let x kind d (erase c body).
Proof.
  intros H. destruct d as [ | [ | | | | ] | | | | | ]; try reflexivity.
  discriminate H.
Qed.

(* What follows the [Let] of a variable, if it is a [Local_allocation]. *)
Definition la_body (body : expr)
    : option (variable * flambda_kind * list simple * expr) :=
  match body with
  | Let y kind' (Prim (Region_primitive Local_allocation) args) body' =>
    Some (y, kind', args, body')
  | _ => None
  end.

Lemma la_body_some (body : expr) (y : variable) (kind' : flambda_kind)
    (args : list simple) (body' : expr) :
  la_body body = Some (y, kind', args, body') ->
  body = Let y kind' (Prim (Region_primitive Local_allocation) args) body'.
Proof.
  destruct body as [y0 k0 d0 b0 | | | | | | | ]; cbn; intros H;
    try discriminate H.
  destruct d0 as [ | [ | | | | [ | | | | ]] | | | | | ]; try discriminate H.
  injection H as -> -> -> ->. reflexivity.
Qed.

Definition la_check (kinds : finite_map flambda_kind) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) : bool :=
  match la_body body with
  | Some (y, kind', args, body') =>
    (is_allocation d || is_cell_allocation d)
    && Pos.eqb y x && kind_eqb kind' kind
    && match args with
       | Var x' :: _ => Pos.eqb x' x
       | _ => false
       end
    && regions_plain ((x, kind) :: (x, kind) :: kinds) body'
  | None => regions_plain ((x, kind) :: kinds) body
  end.

Lemma regions_plain_let (kinds : finite_map flambda_kind) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  region_prim_of d = None ->
  regions_plain kinds (Let x kind d body)
  = kind_not_region kind
    && forallb (simple_not_region kinds) (simples_of_named d)
    && set_plain x d body && la_check kinds x kind d body.
Proof.
  intros H. unfold la_check.
  destruct d as [ | [ | | | | ] | | | | | ]; try discriminate H;
    cbn [regions_plain];
    (destruct body as [y0 k0 d0 b0 | | | | | | | ]; try reflexivity;
     destruct d0 as [ | [ | | | | [ | | | | ]] | | | | | ];
     reflexivity).
Qed.

Definition named_is_set (d : named) : bool :=
  match d with
  | Set_of_closures _ _ | Closure_of_set _ _ _ => true
  | _ => false
  end.

Section Unfold.

Variables (F0 : facts) (sol0 : solution) (b0 : bool)
  (ucids0 : list code_id) (ff0 : positive) (types0 : subkinds_rewriting).

Lemma rebuild_let_pending (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  is_pending_projection en x d = true ->
  rebuild_expr F0 sol0 b0 ucids0 ff0 types0 en (Let x kind d body)
  = Let x kind d
      (rebuild_expr F0 sol0 b0 ucids0 ff0 types0
         (bind_variables_then en [(x, kind)] (tl (env_projections en)))
         body).
Proof. intros H. cbn [rebuild_expr]. rewrite H. reflexivity. Qed.

Lemma rebuild_let_set (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  is_pending_projection en x d = false -> named_is_set d = true ->
  rebuild_expr F0 sol0 b0 ucids0 ff0 types0 en (Let x kind d body)
  = if is_dead_var sol0 x kind
       || existsb (fun y => is_dead_var sol0 y Value)
            (projections_of_set x d body)
    then Invalid
    else if is_var_used sol0 x kind
            || existsb (fun y => is_var_used sol0 y Value)
                 (projections_of_set x d body)
    then
      Let x kind
        (rebuild_set sol0 en (set_closure_nodes F0 (Node_variable x)) d)
        (rebuild_expr F0 sol0 b0 ucids0 ff0 types0
           (bind_variables_then en [(x, kind)]
              (projections_of_set x d body))
           body)
    else
      rebuild_expr F0 sol0 b0 ucids0 ff0 types0 (bind_variables en [(x, kind)])
        body.
Proof.
  intros H Hs. cbn [rebuild_expr]. rewrite H.
  destruct d; try discriminate Hs; reflexivity.
Qed.

Lemma rebuild_let_other (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  is_pending_projection en x d = false -> named_is_set d = false ->
  rebuild_expr F0 sol0 b0 ucids0 ff0 types0 en (Let x kind d body)
  = if is_dead_var sol0 x kind then Invalid
    else if is_var_used sol0 x kind
    then
      Let x kind (rebuild_named sol0 en x d)
        (rebuild_expr F0 sol0 b0 ucids0 ff0 types0
           (bind_variables en [(x, kind)])
           body)
    else
      rebuild_expr F0 sol0 b0 ucids0 ff0 types0 (bind_variables en [(x, kind)])
        body.
Proof.
  intros H Hs. cbn [rebuild_expr]. rewrite H.
  destruct d; try discriminate Hs; reflexivity.
Qed.

Lemma rebuild_let_cont_rec (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  rebuild_expr F0 sol0 b0 ucids0 ff0 types0 en
    (Let_cont_recursive handlers body)
  = Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, rebuild_handler F0 sol0 b0 ucids0 ff0 types0
                  (bind_continuations en
                     (map (fun '(k, h) => (k, handler_mask sol0 false h))
                        handlers))
                  false (handler_mask sol0 false h) h))
         handlers)
      (rebuild_expr F0 sol0 b0 ucids0 ff0 types0
         (bind_continuations en
            (map (fun '(k, h) => (k, handler_mask sol0 false h)) handlers))
         body).
Proof. reflexivity. Qed.

Lemma rebuild_let_cont (en : rebuild_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  rebuild_expr F0 sol0 b0 ucids0 ff0 types0 en
    (Let_cont_non_recursive k h body)
  = if among k
         (free_continuations_of_expr
            (rebuild_expr F0 sol0 b0 ucids0 ff0 types0
               (bind_continuations en
                  [(k, handler_mask sol0 (handler_is_exn_handler h) h)])
               body))
    then
      Let_cont_non_recursive k
        (rebuild_handler F0 sol0 b0 ucids0 ff0 types0 en true
           (handler_mask sol0 (handler_is_exn_handler h) h) h)
        (rebuild_expr F0 sol0 b0 ucids0 ff0 types0
           (bind_continuations en
              [(k, handler_mask sol0 (handler_is_exn_handler h) h)])
           body)
    else
      rebuild_expr F0 sol0 b0 ucids0 ff0 types0
        (bind_continuations en
           [(k, handler_mask sol0 (handler_is_exn_handler h) h)])
        body.
Proof. reflexivity. Qed.

Lemma rebuild_let_static (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  rebuild_expr F0 sol0 b0 ucids0 ff0 types0 en (Let_static bindings body)
  = Let_static
      (flat_map (rebuild_binding F0 sol0 b0 ucids0 ff0 types0 en) bindings)
      (rebuild_expr F0 sol0 b0 ucids0 ff0 types0 (bind_variables en [])
         body).
Proof. reflexivity. Qed.

Lemma rebuild_handler_unfold (en : rebuild_env) (dc : bool)
    (mask : list bool) (params : list variable) (md : handler_metadata)
    (hbody : expr) :
  rebuild_handler F0 sol0 b0 ucids0 ff0 types0 en dc mask
    (Handler params md hbody)
  = Handler (filter_kept mask params)
      (kept_handler_metadata types0 mask params md)
      (if dc
          && existsb (fun '(x, kind) => is_dead_var sol0 x kind)
               (with_kinds params (hm_params_kinds md))
       then Invalid
       else
         rebuild_expr F0 sol0 b0 ucids0 ff0 types0
           (bind_variables en (with_kinds params (hm_params_kinds md)))
           hbody).
Proof. reflexivity. Qed.

Lemma rebuild_binding_code (en : rebuild_env) (id : code_id)
    (params : list variable) (mc : variable) (ret exn : continuation)
    (md : code_metadata) (cbody : expr) :
  rebuild_binding F0 sol0 b0 ucids0 ff0 types0 en
    (Bind_code id (Code params mc ret exn md cbody))
  = if is_code_id_used sol0 id
    then
      [Bind_code id
         (Code params mc ret exn (kept_code_metadata types0 id md)
            (if existsb (fun '(x, kind) => is_dead_var sol0 x kind)
                  ((mc, Value) :: with_kinds params
                                    (unarize (cm_params_arity md)))
             then Invalid
             else
               rebuild_expr F0 sol0 b0 ucids0 ff0 types0
                 (bind_variables FlambdaReaperRebuild.empty_env
                    ((mc, Value)
                     :: with_kinds params (unarize (cm_params_arity md))))
                 cbody))]
    else [Bind_deleted_code id].
Proof. reflexivity. Qed.

End Unfold.

Lemma pending_not_projection (en : rebuild_env) (x : variable) (d : named) :
  (∀ mf mt c, d <> Project_function_slot mf mt c) ->
  is_pending_projection en x d = false.
Proof.
  intros H. unfold is_pending_projection.
  destruct (env_projections en); [ reflexivity | ].
  destruct d; try reflexivity. exfalso. eapply H. reflexivity.
Qed.

Lemma pending_prim (en : rebuild_env) (x : variable) (p : prim)
    (args : list simple) :
  is_pending_projection en x (Prim p args) = false.
Proof. apply pending_not_projection. intros ? ? ? ?. discriminate. Qed.

Lemma pending_simple (en : rebuild_env) (x : variable) (s : simple) :
  is_pending_projection en x (Simple s) = false.
Proof. apply pending_not_projection. intros ? ? ? ?. discriminate. Qed.

Lemma rebuild_named_plain (s0 : solution) (en : rebuild_env) (x : variable)
    (d : named) :
  region_prim_of d = None -> region_prim_of (rebuild_named s0 en x d) = None.
Proof.
  intros H. destruct d as [ | [u | | v | h | ] | | | | | ];
    try discriminate H; try reflexivity.
  - destruct u; try reflexivity. cbn [rebuild_named].
    destruct (field_used s0 _ _); reflexivity.
  - destruct v; reflexivity.
  - destruct h; reflexivity.
Qed.

Lemma rebuild_set_plain (s0 : solution) (en : rebuild_env)
    (bound : list node) (d : named) :
  region_prim_of d = None -> region_prim_of (rebuild_set s0 en bound d) = None.
Proof. intros H. destruct d; try exact H; reflexivity. Qed.

(* ======================================================================== *)
(* The projections of a set                                                 *)
(* ======================================================================== *)

Lemma projected_closures_nil (x : variable) (slot : function_slot)
    (e : expr) :
  projected_closures x slot [] e = [].
Proof. destruct e; reflexivity. Qed.

Lemma projected_closures_erase (x : variable) (slot : function_slot) :
  ∀ (others : list function_slot) (body : expr),
    projections_plain x slot others body = true ->
    projected_closures x slot others (erase [] body)
    = projected_closures x slot others body.
Proof.
  intros others. induction others as [ | o os IH]; intros body H;
    [ rewrite !projected_closures_nil; reflexivity | ].
  destruct body as [y k d b0 | | | | | | | ]; try reflexivity.
  destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
  - apply region_prim_of_some in Hrp. subst d. discriminate H.
  - rewrite (erase_let_plain _ _ _ _ _ Hrp).
    destruct d as [ | | | | | | mf mt c]; try reflexivity.
    destruct c as [x' | | ]; try reflexivity.
    cbn [projections_plain] in H. cbn [projected_closures].
    destruct (Pos.eqb x' x && Pos.eqb mf slot && Pos.eqb mt o
              && negb (Pos.eqb y x));
      [ rewrite (IH b0 H); reflexivity | reflexivity ].
Qed.

Lemma projections_of_set_erase (x : variable) (d : named) (body : expr) :
  set_plain x d body = true ->
  projections_of_set x d (erase [] body) = projections_of_set x d body.
Proof.
  intros H. destruct d; try reflexivity. cbn [set_plain] in H.
  cbn [projections_of_set]. exact (projected_closures_erase _ _ _ _ H).
Qed.

(* The projections still to be rebuilt are those of the chain that the
   term begins with. *)
Definition pending_ok (en : rebuild_env) (e : expr) : Prop :=
  ∃ (x : variable) (slot : function_slot) (others : list function_slot),
    env_projections en = projected_closures x slot others e.

Lemma pending_ok_nil (en : rebuild_env) (e : expr) :
  env_projections en = [] -> pending_ok en e.
Proof.
  intros H. exists 1%positive, 1%positive, []. rewrite H.
  rewrite projected_closures_nil. reflexivity.
Qed.

Lemma pending_ok_let_nil (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  (∀ mf mt c, d <> Project_function_slot mf mt c) ->
  pending_ok en (Let x kind d body) -> env_projections en = [].
Proof.
  intros Hd (x0 & s0 & o0 & H). rewrite H.
  destruct o0; [ reflexivity | ].
  destruct d; try reflexivity. exfalso. eapply Hd. reflexivity.
Qed.

Lemma pending_ok_step (en : rebuild_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  is_pending_projection en x d = true ->
  pending_ok en (Let x kind d body) ->
  pending_ok (bind_variables_then en [(x, kind)] (tl (env_projections en)))
    body.
Proof.
  intros Hp (x0 & s0 & o0 & H). unfold pending_ok. cbn [env_projections
    bind_variables_then]. rewrite H.
  unfold is_pending_projection in Hp. rewrite H in Hp.
  destruct o0 as [ | o os]; [ discriminate Hp | ].
  destruct d as [ | | | | | | mf mt c]; try (cbn in Hp; discriminate Hp).
  destruct c as [x' | | ]; try (cbn in Hp; discriminate Hp).
  cbn [projected_closures].
  destruct (Pos.eqb x' x0 && Pos.eqb mf s0 && Pos.eqb mt o
            && negb (Pos.eqb x x0)).
  - exists x0, s0, os. reflexivity.
  - exists x0, s0, []. rewrite projected_closures_nil. reflexivity.
Qed.

Lemma pending_ok_set (en : rebuild_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  pending_ok (bind_variables_then en [(x, kind)]
                (projections_of_set x d body)) body.
Proof.
  unfold pending_ok. cbn [env_projections bind_variables_then].
  destruct d; try (exists 1%positive, 1%positive, [];
                   rewrite projected_closures_nil; reflexivity).
  cbn [projections_of_set]. eexists _, _, _. reflexivity.
Qed.

(* ======================================================================== *)
(* What the rebuild makes of a call, a jump and a switch is erased to       *)
(* itself                                                                   *)
(* ======================================================================== *)

Lemma erase_rebuild_apply (F0 : facts) (s0 : solution) (b0 : bool)
    (u0 : list code_id) (f0 : positive) (en : rebuild_env)
    (ap : apply_expr) :
  forallb kind_not_region (unarize (ap_return_arity ap)) = true ->
  erase [] (rebuild_apply F0 s0 b0 u0 f0 None en ap)
  = rebuild_apply F0 s0 b0 u0 f0 None en ap.
Proof.
  intros H. unfold rebuild_apply.
  destruct (simple_is_dead s0 en (ap_callee ap)
            || existsb (simple_is_dead s0 en) (ap_args ap));
    [ reflexivity | ].
  unfold make_apply_wrapper. cbn [ap_continuation ap_return_arity].
  destruct (lookup (ap_continuation ap) (env_conts en)) as [mask | ];
    [ | reflexivity ].
  destruct (forallb (fun keep => keep) mask); [ reflexivity | ].
  cbn [erase]. rewrite erase_handler_no_region; [ reflexivity | exact H ].
Qed.

Lemma erase_rebuild_apply_cont (s0 : solution) (en : rebuild_env)
    (ac : apply_cont) :
  erase [] (rebuild_apply_cont s0 en ac) = rebuild_apply_cont s0 en ac.
Proof.
  unfold rebuild_apply_cont. destruct (rewrite_apply_cont s0 en ac);
    reflexivity.
Qed.

Lemma map_arms_id (arms : list (Z * apply_cont)) :
  map (fun '(i, ac) => (i, erase_apply_cont [] ac)) arms = arms.
Proof.
  induction arms as [ | [i ac] arms IH]; [ reflexivity | ].
  cbn [map]. rewrite IH. reflexivity.
Qed.

Lemma erase_rebuild_switch (s0 : solution) (en : rebuild_env)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  erase [] (rebuild_switch s0 en scrutinee arms)
  = rebuild_switch s0 en scrutinee arms.
Proof.
  unfold rebuild_switch.
  match goal with
  |- context [ Keyed.is_empty ?l ] => destruct (Keyed.is_empty l)
  end; [ reflexivity | ].
  cbn [erase]. rewrite map_arms_id. reflexivity.
Qed.

(* ======================================================================== *)
(* Induction with the term after the next                                   *)
(* ======================================================================== *)

Lemma regions_ind (P : expr -> Prop) (Q : continuation_handler -> Prop) :
  (∀ x kind defining_expr body,
     P body ->
     (∀ y kind' d' body', body = Let y kind' d' body' -> P body') ->
     P (Let x kind defining_expr body)) ->
  (∀ bindings body,
     (∀ (id : code_id) (c : code), Bind_code id c ∈ bindings ->
        P (code_body c)) ->
     P body -> P (Let_static bindings body)) ->
  (∀ k h body, Q h -> P body -> P (Let_cont_non_recursive k h body)) ->
  (∀ handlers body,
     (∀ k h, (k, h) ∈ handlers -> Q h) -> P body ->
     P (Let_cont_recursive handlers body)) ->
  (∀ ap, P (Apply ap)) ->
  (∀ ac, P (Apply_cont ac)) ->
  (∀ scrutinee arms, P (Switch scrutinee arms)) ->
  P Invalid ->
  (∀ params metadata handler,
     P handler -> Q (Handler params metadata handler)) ->
  (∀ e, P e) ∧ (∀ h, Q h).
Proof.
  intros Hlet Hstatic Hnonrec Hrec Happly Happly_cont Hswitch Hinvalid
    Hhandler.
  assert (He : ∀ e, P e).
  { fix IH 1 with (IHh (h : continuation_handler) {struct h} : Q h).
    - intros [x kind d body | bindings body | k h body | handlers body | ap
             | ac | scrutinee arms | ].
      + apply Hlet; [ apply IH | ].
        intros y kind' d' body' Heq.
        destruct body as [y0 k0 d0 b0 | | | | | | | ]; try discriminate Heq.
        injection Heq as _ _ _ Hb. rewrite <- Hb. apply IH.
      + apply Hstatic; [ | apply IH ].
        exact
          ((fix go (l : list static_binding)
              : ∀ id c, Bind_code id c ∈ l -> P (code_body c) :=
              match l with
              | [] => fun id c Hin => match Hin with end
              | b :: l' =>
                fun id c Hin =>
                  match Hin with
                  | or_introl Heq =>
                    eq_ind b
                      (fun b0 =>
                         match b0 with
                         | Bind_code _ c0 => P (code_body c0)
                         | _ => True
                         end)
                      (match b as b0
                         return
                           (match b0 with
                            | Bind_code _ c0 => P (code_body c0)
                            | _ => True
                            end)
                       with
                       | Bind_code _ c0 => IH (code_body c0)
                       | _ => I
                       end)
                      (Bind_code id c) Heq
                  | or_intror Hin' => go l' id c Hin'
                  end
              end) bindings).
      + apply Hnonrec; [ apply IHh | apply IH ].
      + apply Hrec; [ | apply IH ].
        exact
          ((fix go (l : list (continuation * continuation_handler))
              : ∀ k h, (k, h) ∈ l -> Q h :=
              match l with
              | [] => fun k h Hin => match Hin with end
              | (k', h') :: l' =>
                fun k h Hin =>
                  match Hin with
                  | or_introl Heq =>
                    eq_ind (k', h') (fun p => Q (snd p)) (IHh h') (k, h) Heq
                  | or_intror Hin' => go l' k h Hin'
                  end
              end) handlers).
      + apply Happly.
      + apply Happly_cont.
      + apply Hswitch.
      + exact Hinvalid.
    - intros [params metadata handler]. apply Hhandler, IH. }
  split; [ exact He | ].
  intros [params metadata handler]. apply Hhandler, He.
Qed.

(* ======================================================================== *)
(* Helpers of the commutation                                               *)
(* ======================================================================== *)

Lemma filter_kept_forallb {A : Type} (f : A -> bool) (mask : list bool)
    (l : list A) :
  forallb f l = true -> forallb f (filter_kept mask l) = true.
Proof.
  revert mask. induction l as [ | a l IH]; intros mask H; [ reflexivity | ].
  cbn in H. apply andb_true_iff in H as [Ha H].
  destruct mask as [ | keep mask]; cbn [filter_kept].
  - cbn [forallb]. rewrite Ha, (IH [] H). reflexivity.
  - destruct keep; [ cbn [forallb]; rewrite Ha | ]; apply IH; exact H.
Qed.

Lemma kept_metadata_plain (mask : list bool) (params : list variable)
    (md : handler_metadata) :
  forallb kind_not_region (hm_params_kinds md) = true ->
  forallb kind_not_region
    (hm_params_kinds (kept_handler_metadata None mask params md)) = true.
Proof.
  intros H. cbn [kept_handler_metadata]. unfold with_params_kinds.
  destruct (hm_is_exn_handler md); apply filter_kept_forallb; exact H.
Qed.

Lemma handler_mask_erase (s0 : solution) (exn : bool)
    (h : continuation_handler) (c : region_conts) :
  forallb kind_not_region (handler_kinds h) = true ->
  handler_mask s0 exn (erase_handler c h) = handler_mask s0 exn h.
Proof.
  destruct h as [params md hbody]. intros H. cbn [handler_kinds] in H.
  rewrite (erase_handler_no_region c params md hbody H). reflexivity.
Qed.

Lemma handler_exn_erase (h : continuation_handler) (c : region_conts) :
  forallb kind_not_region (handler_kinds h) = true ->
  handler_is_exn_handler (erase_handler c h) = handler_is_exn_handler h.
Proof.
  destruct h as [params md hbody]. intros H. cbn [handler_kinds] in H.
  rewrite (erase_handler_no_region c params md hbody H). reflexivity.
Qed.

Lemma handler_kinds_plain (kinds : finite_map flambda_kind)
    (h : continuation_handler) :
  handler_regions_plain kinds h = true ->
  forallb kind_not_region (handler_kinds h) = true.
Proof.
  destruct h as [params md hbody]. cbn [handler_regions_plain handler_kinds].
  intros H. apply andb_true_iff in H as [H _]. exact H.
Qed.

(* The places of the regions of a handler that the rebuild makes. *)
Lemma rebuilt_handler_places (F0 : facts) (s0 : solution) (b0 : bool)
    (u0 : list code_id) (f0 : positive) (en : rebuild_env) (dc : bool)
    (mask : list bool) (h : continuation_handler) :
  forallb kind_not_region (handler_kinds h) = true ->
  forallb negb
    (region_params (rebuild_handler F0 s0 b0 u0 f0 None en dc mask h))
  = true.
Proof.
  destruct h as [params md hbody]. intros H. cbn [handler_kinds] in H.
  cbn [rebuild_handler region_params handler_kinds].
  apply region_places_plain, kept_metadata_plain, H.
Qed.

Lemma set_plain_region_head (x : variable) (d : named) (y : variable)
    (k : flambda_kind) (p : region_primitive) (args : list simple)
    (b' : expr) :
  set_plain x d (Let y k (Prim (Region_primitive p) args) b') = true ->
  ∀ e' : expr, projections_of_set x d e' = [].
Proof.
  intros H e'. destruct d; try reflexivity. cbn [set_plain] in H.
  cbn [projections_of_set].
  destruct (other_slots slot function_decls) as [ | o os];
    [ apply projected_closures_nil | discriminate H ].
Qed.

Lemma allocation_not_projection (d : named) :
  is_allocation d || is_cell_allocation d = true ->
  ∀ mf mt c, d <> Project_function_slot mf mt c.
Proof. intros H mf mt c ->. discriminate H. Qed.

Lemma kind_of_bound (en : rebuild_env) (x : variable) (kind : flambda_kind) :
  kind_of_variable (bind_variables en [(x, kind)]) x = kind.
Proof.
  unfold kind_of_variable. cbn. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma pending_ok_bind (en : rebuild_env)
    (bound : list (variable * flambda_kind))
    (e : expr) :
  pending_ok (bind_variables en bound) e.
Proof. apply pending_ok_nil. reflexivity. Qed.


Lemma masks_erase (s0 : solution) (c : region_conts)
    (hs : list (continuation * continuation_handler)) :
  (∀ k h, (k, h) ∈ hs -> forallb kind_not_region (handler_kinds h) = true) ->
  map (fun '(k, h) => (k, handler_mask s0 false h))
    (map (fun '(k, h) => (k, erase_handler c h)) hs)
  = map (fun '(k, h) => (k, handler_mask s0 false h)) hs.
Proof.
  induction hs as [ | [k h] hs IH]; intros H; [ reflexivity | ].
  cbn [map]. rewrite (handler_mask_erase s0 false h c (H k h (or_introl
    eq_refl))).
  rewrite IH; [ reflexivity | ]. intros k' h' Hin. apply (H k'). right.
  exact Hin.
Qed.

Lemma conts_plain_map {X : Type} (f : X -> continuation * list bool)
    (l : list X) :
  (∀ x, x ∈ l -> forallb negb (snd (f x)) = true) ->
  conts_plain (map f l ++ []).
Proof.
  intros H. apply conts_plain_app; [ | apply conts_plain_nil ].
  intros k p Hin. apply in_map_iff in Hin as [x [Heq Hin]].
  specialize (H x Hin). rewrite Heq in H. exact H.
Qed.

Lemma rec_conts_plain (hs : list (continuation * continuation_handler)) :
  (∀ k h, (k, h) ∈ hs -> forallb kind_not_region (handler_kinds h) = true) ->
  conts_plain (map (fun '(k, h) => (k, region_params h)) hs ++ []).
Proof.
  intros H. apply conts_plain_app; [ | apply conts_plain_nil ].
  intros k p Hin. apply in_map_iff in Hin as [[k' h] [Heq Hin]].
  injection Heq as <- <-. apply region_places_plain. exact (H k' h Hin).
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

Abbreviation RB := (rebuild_expr F sol b ucids ff None).
Abbreviation RB' := (rebuild_expr F' sol' b ucids ff None).

Lemma masks_eq (hs : list (continuation * continuation_handler)) :
  map (fun '(k, h) => (k, handler_mask sol' false h)) hs
  = map (fun '(k, h) => (k, handler_mask sol false h)) hs.
Proof.
  induction hs as [ | [k h] hs IH]; [ reflexivity | ].
  cbn [map]. rewrite handler_mask_eq, IH. reflexivity.
Qed.

Lemma rebuilt_conts_plain (en : rebuild_env)
    (hs : list (continuation * continuation_handler)) :
  (∀ k h, (k, h) ∈ hs -> forallb kind_not_region (handler_kinds h) = true) ->
  conts_plain
    (map (fun '(k, h) => (k, region_params h))
       (map (fun '(k, h) =>
               (k, rebuild_handler F sol b ucids ff None en false
                     (handler_mask sol false h) h)) hs) ++ []).
Proof.
  intros H. apply conts_plain_app; [ | apply conts_plain_nil ].
  intros k p Hin. apply in_map_iff in Hin as [[k' h'] [Heq Hin]].
  injection Heq as <- <-.
  apply in_map_iff in Hin as [[k0 h0] [Heq Hin]].
  injection Heq as <- <-. apply rebuilt_handler_places.
  exact (H k0 h0 Hin).
Qed.

Lemma rebuild_erase_commutes_gen :
  (∀ e : expr, ∀ en en' : rebuild_env,
     regions_plain (env_kinds en) e = true -> env_rel en en' ->
     pending_ok en e ->
     erase [] (RB en e) = RB' en' (erase [] e))
  ∧ (∀ h : continuation_handler,
       ∀ (en en' : rebuild_env) (dc : bool) (mask : list bool),
         handler_regions_plain (env_kinds en) h = true -> env_rel en en' ->
         erase_handler [] (rebuild_handler F sol b ucids ff None en dc mask h)
         = rebuild_handler F' sol' b ucids ff None en' dc mask
             (erase_handler [] h)).
Proof.
  apply regions_ind.
  - (* Let *)
    intros x kind d body IH IH2 en en' Hpl Hen Hpend.
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d.
      assert (Hnil : env_projections en = [])
        by (apply (pending_ok_let_nil en x kind
                     (Prim (Region_primitive p) args) body);
            [ intros ? ? ? ?; discriminate | exact Hpend ]).
      rewrite (rebuild_let_other F sol b ucids ff None en x kind _ body
                 (pending_prim _ _ _ _) eq_refl).
      destruct p; destruct args as [ | s [ | s' args]];
        cbn [regions_plain] in Hpl; try discriminate Hpl.
      * (* Begin_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        cbn [is_dead_var is_var_used rebuild_named map erase
               erase_region_primitive].
        apply IH; [ exact Hb | apply env_rel_bind_region; assumption
                  | apply pending_ok_bind ].
      * (* Begin_try_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        cbn [is_dead_var is_var_used rebuild_named map erase
               erase_region_primitive].
        apply IH; [ exact Hb | apply env_rel_bind_region; assumption
                  | apply pending_ok_bind ].
      * (* End_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        cbn [erase erase_region_primitive].
        rewrite (rebuild_let_other F' sol' b ucids ff None en' x kind
                   (Simple unit_simple) _ (pending_simple _ _ _) eq_refl).
        rewrite dead_var_eq, var_used_eq.
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- cbn [rebuild_named map erase erase_region_primitive rewrite_simple
                  unit_simple].
           f_equal. apply IH; [ exact Hb | apply env_rel_bind, Hen
                              | apply pending_ok_bind ].
        -- apply IH; [ exact Hb | apply env_rel_bind, Hen
                     | apply pending_ok_bind ].
      * (* End_try_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        cbn [erase erase_region_primitive].
        rewrite (rebuild_let_other F' sol' b ucids ff None en' x kind
                   (Prim (Unary (Opaque_identity Value)) [unit_simple]) _
                   (pending_prim _ _ _ _) eq_refl).
        rewrite dead_var_eq, var_used_eq.
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- cbn [rebuild_named map erase erase_region_primitive rewrite_simple
                  unit_simple].
           f_equal. apply IH; [ exact Hb | apply env_rel_bind, Hen
                              | apply pending_ok_bind ].
        -- apply IH; [ exact Hb | apply env_rel_bind, Hen
                     | apply pending_ok_bind ].
    + (* Not a primitive of regions. *)
      rewrite (regions_plain_let _ _ _ _ _ Hrp) in Hpl.
      apply andb_true_iff in Hpl as [Hpl Hla].
      apply andb_true_iff in Hpl as [Hpl Hset].
      apply andb_true_iff in Hpl as [Hk Hd].
      destruct Hen as (Hc & Hps & Hkinds).
      assert (Hen : env_rel en en') by (split; [ | split ]; assumption).
      rewrite (erase_let_plain _ _ _ _ _ Hrp).
      unfold la_check in Hla.
      destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab.
      * (* A local allocation. *)
        apply la_body_some in Hlab. subst body.
        apply andb_true_iff in Hla as [Hla Hb'].
        apply andb_true_iff in Hla as [Hla Hargs].
        apply andb_true_iff in Hla as [Hla Hk'].
        apply andb_true_iff in Hla as [Halloc Hy].
        apply Pos.eqb_eq in Hy. subst y.
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk'. subst k'.
        destruct args as [ | [x' | | ] rs]; try discriminate Hargs.
        apply Pos.eqb_eq in Hargs. subst x'.
        cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
        pose proof (allocation_not_projection d Halloc) as Hnproj.
        (* The two [Let]s of [x] are kept or removed together. *)
        assert (Hinner :
          ∀ en1 : rebuild_env,
            env_kinds en1 = (x, kind) :: env_kinds en ->
            env_projections en1 = [] ->
            env_conts en1 = env_conts en ->
            is_dead_var sol x kind = false ->
            erase []
              (RB en1
                 (Let x kind
                    (Prim (Region_primitive Local_allocation) (Var x :: rs))
                    body'))
            = RB' (bind_variables en' [(x, kind)]) (erase [] body')).
        { intros en1 Hk1 Hp1 Hc1 Hnd.
          rewrite (rebuild_let_other F sol b ucids ff None en1 x kind
                     (Prim (Region_primitive Local_allocation) (Var x :: rs))
                     body' (pending_prim _ _ _ _) eq_refl).
          rewrite Hnd.
          assert (Hrel : env_rel (bind_variables en1 [(x, kind)])
                           (bind_variables en' [(x, kind)])).
          { split; [ cbn; rewrite Hc1, Hc; reflexivity | ].
            split; [ reflexivity | ].
            intros z Hz. unfold region_variable in Hz.
            cbn [bind_variables bind_variables_then env_kinds rev app lookup]
              in Hz |- *.
            rewrite Hk1 in Hz |- *. cbn [lookup] in Hz |- *.
            destruct (Pos.eqb z x); [ reflexivity | exact (Hkinds z Hz) ]. }
          assert (Hb'' : regions_plain
                           (env_kinds (bind_variables en1 [(x, kind)]))
                           body' = true)
            by (cbn [bind_variables bind_variables_then env_kinds rev app];
                rewrite Hk1; exact Hb').
          destruct (is_var_used sol x kind) eqn:Hu1.
          - cbn [rebuild_named map erase erase_region_primitive].
            assert (Hrw : rewrite_simple sol en1 (Var x) = Var x).
            { cbn [rewrite_simple]. unfold kind_of_variable. rewrite Hk1.
              cbn [lookup]. rewrite Pos.eqb_refl, Hu1. reflexivity. }
            rewrite Hrw. cbn [erase erase_region_primitive].
            rewrite Pos.eqb_refl.
            apply IH2 with (y := x) (kind' := kind)
              (d' := Prim (Region_primitive Local_allocation) (Var x :: rs));
              [ reflexivity | exact Hb'' | exact Hrel | apply pending_ok_bind ].
          - apply IH2 with (y := x) (kind' := kind)
              (d' := Prim (Region_primitive Local_allocation) (Var x :: rs));
              [ reflexivity | exact Hb'' | exact Hrel
              | apply pending_ok_bind ]. }
        destruct (named_is_set d) eqn:Hs.
        -- rewrite (rebuild_let_set F sol b ucids ff None en x kind d _
                      (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite (rebuild_let_set F' sol' b ucids ff None en' x kind d _
                      (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite !(set_plain_region_head _ _ _ _ _ _ _ Hset).
           cbn [existsb]. rewrite !orb_false_r, dead_var_eq, var_used_eq.
           destruct (is_dead_var sol x kind) eqn:Hdead; [ reflexivity | ].
           destruct (is_var_used sol x kind) eqn:Hused.
           ++ rewrite (erase_let_plain _ _ _ _ _
                         (rebuild_set_plain _ _ _ _ Hrp)).
              rewrite (rebuild_set_eq _ _ Hen _ _ Hd).
              unfold set_closure_nodes. rewrite HC. f_equal.
              apply (Hinner (bind_variables_then en [(x, kind)] []));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
           ++ apply (Hinner (bind_variables en [(x, kind)]));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
        -- rewrite (rebuild_let_other F sol b ucids ff None en x kind d _
                      (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite (rebuild_let_other F' sol' b ucids ff None en' x kind d _
                      (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite dead_var_eq, var_used_eq.
           destruct (is_dead_var sol x kind) eqn:Hdead; [ reflexivity | ].
           destruct (is_var_used sol x kind) eqn:Hused.
           ++ rewrite (erase_let_plain _ _ _ _ _
                         (rebuild_named_plain _ _ _ _ Hrp)).
              rewrite (rebuild_named_eq _ _ Hen _ _ Hd). f_equal.
              apply (Hinner (bind_variables en [(x, kind)]));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
           ++ apply (Hinner (bind_variables en [(x, kind)]));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
      * (* No local allocation follows. *)
        destruct (is_pending_projection en x d) eqn:Hp.
        -- assert (Hp' : is_pending_projection en' x d = true)
             by (unfold is_pending_projection in *; rewrite Hps; exact Hp).
           rewrite (rebuild_let_pending F sol b ucids ff None en x kind d body
                      Hp).
           rewrite (rebuild_let_pending F' sol' b ucids ff None en' x kind d
                      (erase [] body) Hp').
           rewrite (erase_let_plain _ _ _ _ _ Hrp). f_equal.
           rewrite Hps. apply IH;
             [ exact Hla | apply env_rel_bind_then, Hen
             | exact (pending_ok_step _ _ _ _ _ Hp Hpend) ].
        -- assert (Hp' : is_pending_projection en' x d = false)
             by (unfold is_pending_projection in *; rewrite Hps; exact Hp).
           destruct (named_is_set d) eqn:Hs.
           ++ rewrite (rebuild_let_set F sol b ucids ff None en x kind d body
                         Hp Hs).
              rewrite (rebuild_let_set F' sol' b ucids ff None en' x kind d
                         (erase [] body) Hp' Hs).
              rewrite (projections_of_set_erase _ _ _ Hset).
              rewrite dead_var_eq.
              rewrite (existsb_ext_in (fun y => is_dead_var sol' y Value)
                         (fun y => is_dead_var sol y Value))
                by (intros; apply dead_var_eq).
              rewrite var_used_eq.
              rewrite (existsb_ext_in (fun y => is_var_used sol' y Value)
                         (fun y => is_var_used sol y Value))
                by (intros; apply var_used_eq).
              destruct (_ || _); [ reflexivity | ].
              destruct (_ || _).
              ** rewrite (erase_let_plain _ _ _ _ _
                            (rebuild_set_plain _ _ _ _ Hrp)).
                 rewrite (rebuild_set_eq _ _ Hen _ _ Hd).
                 unfold set_closure_nodes. rewrite HC. f_equal.
                 apply IH; [ exact Hla | apply env_rel_bind_then, Hen
                           | apply pending_ok_set ].
              ** apply IH; [ exact Hla | apply env_rel_bind, Hen
                           | apply pending_ok_bind ].
           ++ rewrite (rebuild_let_other F sol b ucids ff None en x kind d
                         body Hp Hs).
              rewrite (rebuild_let_other F' sol' b ucids ff None en' x kind d
                         (erase [] body) Hp' Hs).
              rewrite dead_var_eq, var_used_eq.
              destruct (is_dead_var sol x kind); [ reflexivity | ].
              destruct (is_var_used sol x kind).
              ** rewrite (erase_let_plain _ _ _ _ _
                            (rebuild_named_plain _ _ _ _ Hrp)).
                 rewrite (rebuild_named_eq _ _ Hen _ _ Hd). f_equal.
                 apply IH; [ exact Hla | apply env_rel_bind, Hen
                           | apply pending_ok_bind ].
              ** apply IH; [ exact Hla | apply env_rel_bind, Hen
                           | apply pending_ok_bind ].
  - (* Let_static *)
    intros bindings body Hcodes IH en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hbs Hb].
    rewrite !rebuild_let_static. cbn [erase].
    rewrite rebuild_let_static. f_equal.
    + clear IH Hb Hpend.
      induction bindings as [ | bd bds IHb]; [ reflexivity | ].
      cbn [forallb] in Hbs. apply andb_true_iff in Hbs as [Hbd Hbs].
      cbn [flat_map map]. rewrite map_app. f_equal.
      * destruct bd as [id c | id | sym const].
        -- destruct c as [params mc ret exn md cbody].
           cbn [erase_binding erase_code].
           rewrite !rebuild_binding_code, code_id_used_eq.
           destruct (is_code_id_used sol id); cbn [map]; [ | reflexivity ].
           cbn [erase_code erase_binding]. do 3 f_equal.
           rewrite (existsb_ext_in (fun '(x, kind) => is_dead_var sol' x kind)
                      (fun '(x, kind) => is_dead_var sol x kind))
             by (intros [x k] _; apply dead_var_eq).
           match goal with
           |- context [ if existsb ?f ?l then _ else _ ] =>
             destruct (existsb f l)
           end; [ reflexivity | ].
           cbn [binding_regions_plain code_regions_plain] in Hbd.
           apply andb_true_iff in Hbd as [_ Hcb].
           apply (Hcodes id (Code params mc ret exn md cbody)
                    (or_introl eq_refl));
             [ | apply env_rel_refl | apply pending_ok_bind ].
           cbn [code_body bind_variables bind_variables_then env_kinds rev
                  FlambdaReaperRebuild.empty_env].
           rewrite app_nil_r. exact Hcb.
        -- reflexivity.
        -- cbn [rebuild_binding erase_binding].
           rewrite static_const_is_used_eq.
           destruct (static_const_is_used sol sym const); [ | reflexivity ].
           cbn [map erase_binding]. cbn [binding_regions_plain] in Hbd.
           rewrite (rebuild_static_const_eq _ _ Hen _ _ Hbd). reflexivity.
      * apply IHb; [ | exact Hbs ].
        intros id c Hin. apply (Hcodes id c). right. exact Hin.
    + apply IH; [ exact Hb | apply env_rel_bind, Hen | apply pending_ok_bind ].
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hh Hb].
    pose proof (handler_kinds_plain _ _ Hh) as Hhk.
    cbn [erase].
    rewrite (erase_plain _ body
               (conts_plain_cons _ _ _ (region_places_plain _ Hhk)
                  conts_plain_nil)).
    rewrite !rebuild_let_cont.
    rewrite (handler_exn_erase h [] Hhk), (handler_mask_erase _ _ h [] Hhk),
      handler_mask_eq.
    set (mask := handler_mask sol (handler_is_exn_handler h) h).
    assert (IHb : erase [] (RB (bind_continuations en [(k, mask)]) body)
                  = RB' (bind_continuations en' [(k, mask)]) (erase [] body))
      by (apply IH; [ exact Hb | apply env_rel_bind_continuations, Hen
                    | apply pending_ok_nil; reflexivity ]).
    rewrite <- IHb, (proj1 free_continuations_erase).
    destruct (among k (free_continuations_of_expr
                         (RB (bind_continuations en [(k, mask)]) body)));
      [ | reflexivity ].
    cbn [erase]. f_equal.
    + apply IHh; assumption.
    + apply erase_plain, conts_plain_cons; [ | apply conts_plain_nil ].
      apply rebuilt_handler_places, Hhk.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hhs Hb].
    rewrite forallb_forall in Hhs.
    assert (Hplain : ∀ k h, (k, h) ∈ handlers ->
                       forallb kind_not_region (handler_kinds h) = true)
      by (intros k h Hin; exact (handler_kinds_plain _ _ (Hhs (k, h) Hin))).
    rewrite rebuild_let_cont_rec. cbn [erase]. rewrite rebuild_let_cont_rec.
    rewrite (masks_erase sol' _ handlers Hplain).
    rewrite (masks_eq handlers).
    set (en1 := bind_continuations en
                  (map (fun '(k, h) => (k, handler_mask sol false h))
                     handlers)).
    set (en1' := bind_continuations en'
                   (map (fun '(k, h) => (k, handler_mask sol false h))
                      handlers)).
    assert (Hen1 : env_rel en1 en1')
      by (apply env_rel_bind_continuations, Hen).
    f_equal.
    + rewrite !map_map. apply map_ext_in.
      intros [k h] Hin. cbn beta iota.
      rewrite (erase_handler_plain _ h)
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply region_places_plain; exact (Hplain k0 h0 Hin0)).
      rewrite (erase_handler_plain _ (rebuild_handler _ _ _ _ _ _ _ _ _ _))
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply rebuilt_handler_places; exact (Hplain k0 h0 Hin0)).
      rewrite (handler_mask_erase _ _ h [] (Hplain k h Hin)), handler_mask_eq.
      f_equal. apply IHh with (k := k); [ exact Hin | | exact Hen1 ].
      exact (Hhs (k, h) Hin).
    + rewrite (erase_plain _ body) by (apply rec_conts_plain, Hplain).
      rewrite (erase_plain _ (RB en1 body))
        by (apply rebuilt_conts_plain, Hplain).
      apply IH; [ exact Hb | exact Hen1 | apply pending_ok_nil; reflexivity ].
  - (* Apply *)
    intros ap en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hs Hr].
    cbn [rebuild_expr erase]. rewrite (erase_rebuild_apply _ _ _ _ _ _ _ Hr).
    symmetry. exact (rebuild_apply_eq en en' Hen ap Hs).
  - (* Apply_cont *)
    intros ac en en' Hpl Hen Hpend. cbn [regions_plain] in Hpl.
    cbn [rebuild_expr erase]. rewrite erase_rebuild_apply_cont,
      erase_apply_cont_nil.
    symmetry. exact (rebuild_apply_cont_eq en en' Hen ac Hpl).
  - (* Switch *)
    intros scrutinee arms en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hs Harms].
    cbn [rebuild_expr erase]. rewrite erase_rebuild_switch, map_arms_id.
    symmetry. exact (rebuild_switch_eq en en' Hen scrutinee arms Hs Harms).
  - (* Invalid *)
    reflexivity.
  - (* Handler *)
    intros params md hbody IH en en' dc mask Hpl Hen.
    cbn [handler_regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hk Hb].
    rewrite (erase_handler_no_region [] params md hbody Hk).
    rewrite !rebuild_handler_unfold.
    rewrite erase_handler_no_region by (apply kept_metadata_plain, Hk).
    f_equal.
    rewrite (existsb_ext_in (fun '(x, kind) => is_dead_var sol' x kind)
               (fun '(x, kind) => is_dead_var sol x kind))
      by (intros [x k] _; apply dead_var_eq).
    destruct (dc && _); [ reflexivity | ].
    apply IH; [ exact Hb | apply env_rel_bind, Hen | apply pending_ok_bind ].
Qed.

End Commute.
