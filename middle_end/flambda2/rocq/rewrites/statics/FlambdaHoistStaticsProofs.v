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

(* Proofs of the statements of FlambdaHoistStaticsSpec.v. Nothing here
   needs review.

   The unit and the unit with its closed static bindings first are run
   side by side, in step, with the same values: the second state is the
   first with those bindings taken out of its terms, all the code bound,
   and the symbols that were taken out bound beside those that the first
   state has bound and that were not ([hoist_state]). Three things are
   kept along the run of the first. No static name is bound twice (the
   invariant of FlambdaWellFormedProofs.v, for code ids and for
   symbols). What the terms of the state still bind is what the second
   state has bound, where it is taken out, and is not a name taken out
   otherwise ([known_inv]). And every static name that the state holds
   or mentions is bound (the invariant of FlambdaStaticsInScopeProofs.v):
   so whatever the first run looks up, the second finds bound to the
   same thing, and what the second has bound early is not looked at
   before the first binds it. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaUnusedSlotsProofs FlambdaStaticsInScopeSpec
  FlambdaStaticsInScopeProofs FlambdaImportsSpec FlambdaUnitFromProofs
  FlambdaHoistStaticsSpec FlambdaEquivarianceTactic.
From OxCamlFlambda2 Require FlambdaClosuresProofs FlambdaEquivarianceProofs
  FlambdaHoistCodeProofs.
Import ListNotations.

(* ======================================================================== *)
(* Nothing to take out                                                      *)
(* ======================================================================== *)

Lemma no_binding_hoisted (bindings : list static_binding) :
  existsb binding_binds bindings = false ->
  filter is_hoisted bindings = []
  ∧ filter (fun binding => negb (is_hoisted binding)) bindings = bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ split; reflexivity | ].
  cbn [existsb]. intros H. apply orb_false_iff in H as [Hbinding Hrest].
  destruct (IH Hrest) as [IHhoisted IHrest].
  destruct binding as [id c | id | sym const]; simpl in Hbinding;
    try discriminate.
  cbn [filter is_hoisted negb]. rewrite IHhoisted, IHrest.
  split; reflexivity.
Qed.

(* An expression that binds nothing static has nothing to take out. *)
Fixpoint without_hoisted_nothing (e : expr) {struct e} :
  binds_statics e = false -> without_hoisted e = e ∧ hoisted_bindings e = []
with handler_without_hoisted_nothing (h : continuation_handler) {struct h} :
  handler_binds_statics h = false ->
  handler_without_hoisted h = h ∧ handler_hoisted_bindings h = [].
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [binds_statics without_hoisted hoisted_bindings];
      try (intros _; split; reflexivity).
    + intros H. destruct (without_hoisted_nothing body H) as [-> ->].
      split; reflexivity.
    + intros H. apply orb_false_iff in H as [Hbindings Hbody].
      destruct (without_hoisted_nothing body Hbody) as [-> ->].
      destruct (no_binding_hoisted bindings Hbindings) as [-> ->].
      split; reflexivity.
    + intros H. apply orb_false_iff in H as [Hh Hbody].
      destruct (handler_without_hoisted_nothing h Hh) as [-> ->].
      destruct (without_hoisted_nothing body Hbody) as [-> ->].
      split; reflexivity.
    + intros H. apply orb_false_iff in H as [_ Hbody].
      destruct (without_hoisted_nothing body Hbody) as [-> ->].
      split; reflexivity.
  - destruct h as [params metadata handler].
    cbn [handler_binds_statics handler_without_hoisted
         handler_hoisted_bindings].
    intros H. destruct (without_hoisted_nothing handler H) as [-> ->].
    split; reflexivity.
Qed.

Lemma handlers_without_hoisted_nothing
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics handlers = false ->
  ∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers ->
    handler_without_hoisted h = h ∧ handler_hoisted_bindings h = [].
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k' h'] handlers IH]; simpl;
    [ intros _ k h [] | ].
  intros H k h Hin. apply orb_false_iff in H as [Hh Hrest].
  destruct Hin as [Heq | Hin]; [ | exact (IH Hrest k h Hin) ].
  injection Heq as _ <-. exact (handler_without_hoisted_nothing h' Hh).
Qed.

Fixpoint hoisted_bindings_are_hoisted (e : expr) {struct e} :
  forallb is_hoisted (hoisted_bindings e) = true
with handler_hoisted_bindings_are_hoisted (h : continuation_handler)
    {struct h} :
  forallb is_hoisted (handler_hoisted_bindings h) = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [hoisted_bindings]; try reflexivity.
    + apply hoisted_bindings_are_hoisted.
    + rewrite forallb_app, (hoisted_bindings_are_hoisted body), andb_true_r.
      apply forallb_forall. intros binding Hin.
      apply filter_In in Hin. exact (proj2 Hin).
    + rewrite forallb_app, (hoisted_bindings_are_hoisted body),
        (handler_hoisted_bindings_are_hoisted h).
      reflexivity.
    + apply hoisted_bindings_are_hoisted.
  - destruct h as [params metadata handler]. cbn [handler_hoisted_bindings].
    apply hoisted_bindings_are_hoisted.
Qed.

(* ======================================================================== *)
(* A constant that has a value in the empty environment                     *)
(* ======================================================================== *)

Lemma eval_simple_closed (vars : env) (s : simple) (v : value) :
  eval_simple [] s = Some v -> eval_simple vars s = Some v.
Proof.
  destruct s as [x | sym | c]; simpl; [ discriminate | | ];
    intros H; exact H.
Qed.

Lemma eval_simples_closed (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples [] args = Some vs -> eval_simples vars args = Some vs.
Proof.
  induction args as [ | s args IH]; intros vs; simpl;
    [ intros H; exact H | ].
  destruct (eval_simple [] s) as [v | ] eqn:Hs; [ | discriminate ].
  rewrite (eval_simple_closed vars s v Hs).
  destruct (eval_simples [] args) as [rest | ]; [ | discriminate ].
  rewrite (IH rest eq_refl). intros H; exact H.
Qed.

Lemma eval_value_slots_closed (vars : env)
    (slots : list (value_slot * simple)) :
  ∀ vs : list (value_slot * value),
    eval_value_slots [] slots = Some vs ->
    eval_value_slots vars slots = Some vs.
Proof.
  induction slots as [ | [slot s] slots IH]; intros vs; simpl;
    [ intros H; exact H | ].
  destruct (eval_simple [] s) as [v | ] eqn:Hs; [ | discriminate ].
  rewrite (eval_simple_closed vars s v Hs).
  destruct (eval_value_slots [] slots) as [rest | ]; [ | discriminate ].
  rewrite (IH rest eq_refl). intros H; exact H.
Qed.

Lemma eval_static_const_closed (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  eval_static_const [] const = Some allocated ->
  eval_static_const vars const = Some allocated.
Proof.
  destruct const as [code slots | slot syms decls slots | tag fields
                    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots [] slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    rewrite (eval_value_slots_closed vars slots vs Hslots).
    intros H; exact H.
  - destruct (lookup slot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots [] slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    rewrite (eval_value_slots_closed vars slots vs Hslots).
    intros H; exact H.
  - destruct (eval_simples [] fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    rewrite (eval_simples_closed vars fields vs Hfields). intros H; exact H.
  - destruct (eval_simple [] contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    rewrite (eval_simple_closed vars contents v Hcontents).
    intros H; exact H.
  - intros H; exact H.
  - destruct (eval_simples [] elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    rewrite (eval_simples_closed vars elements vs Helements).
    intros H; exact H.
Qed.

(* ======================================================================== *)
(* Lookups                                                                  *)
(* ======================================================================== *)

Lemma lookup_filter_kept {A : Type} (keep : positive -> bool)
    (name : positive) (bindings : list (positive * A)) :
  keep name = true ->
  lookup name (filter (fun binding => keep (fst binding)) bindings)
  = lookup name bindings.
Proof.
  intros Hkeep. induction bindings as [ | [bound a] bindings IH];
    [ reflexivity | ].
  cbn [filter fst]. destruct (keep bound) eqn:Hbound; simpl.
  - rewrite IH. reflexivity.
  - destruct (Pos.eqb_spec name bound) as [-> | _]; [ | exact IH ].
    rewrite Hkeep in Hbound. discriminate.
Qed.

Lemma lookup_filter_dropped {A : Type} (keep : positive -> bool)
    (name : positive) (bindings : list (positive * A)) :
  keep name = false ->
  lookup name (filter (fun binding => keep (fst binding)) bindings) = None.
Proof.
  intros Hkeep. induction bindings as [ | [bound a] bindings IH];
    [ reflexivity | ].
  cbn [filter fst]. destruct (keep bound) eqn:Hbound; simpl; [ | exact IH ].
  destruct (Pos.eqb_spec name bound) as [-> | _]; [ | exact IH ].
  rewrite Hkeep in Hbound. discriminate.
Qed.

(* ======================================================================== *)
(* The group that is bound first                                            *)
(* ======================================================================== *)

(* Binding a list of bindings that are taken out, none of whose names
   is bound and whose names are distinct, binds each code id of the
   list to its code, each symbol to the value of its constant in the
   empty environment, and no other name. *)
Lemma bind_hoisted_from (hs : list static_binding) :
  ∀ statics : static_env,
    forallb is_hoisted hs = true ->
    distinct (flat_map code_ids_of_binding hs) = true ->
    distinct (flat_map symbols_of_binding hs) = true ->
    (∀ id : code_id,
      among id (flat_map code_ids_of_binding hs) = true ->
      lookup id (static_code statics) = None) ->
    (∀ s : symbol,
      among s (flat_map symbols_of_binding hs) = true ->
      lookup s (static_values statics) = None) ->
    ∃ (bound_code : list (code_id * code))
      (bound_values : list (symbol * statically_allocated_value)),
      eval_static_bindings [] hs statics
      = Some (Static_env bound_code bound_values)
      ∧ (∀ (id : code_id) (c : code),
          Bind_code id c ∈ hs -> lookup id bound_code = Some c)
      ∧ (∀ (id : code_id) (c : code),
          lookup id (static_code statics) = Some c ->
          lookup id bound_code = Some c)
      ∧ (∀ id : code_id,
          lookup id bound_code <> None ->
          among id (flat_map code_ids_of_binding hs) = true
          ∨ lookup id (static_code statics) <> None)
      ∧ (∀ (s : symbol) (const : static_const),
          Bind_symbol s const ∈ hs ->
          ∃ allocated : statically_allocated_value,
            eval_static_const [] const = Some allocated
            ∧ lookup s bound_values = Some allocated)
      ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
          lookup s (static_values statics) = Some allocated ->
          lookup s bound_values = Some allocated)
      ∧ (∀ s : symbol,
          lookup s bound_values <> None ->
          among s (flat_map symbols_of_binding hs) = true
          ∨ lookup s (static_values statics) <> None).
Proof.
  induction hs as [ | [id c | id | sym const] hs IH];
    intros statics Hhoisted Hcodes Hsymbols Hfresh_code Hfresh_symbols;
    simpl in Hhoisted, Hcodes, Hsymbols; try discriminate;
    cbn [eval_static_bindings].
  - exists (static_code statics), (static_values statics).
    split; [ destruct statics; reflexivity | ].
    split; [ intros id c [] | ].
    split; [ auto | ].
    split; [ intros id H; right; exact H | ].
    split; [ intros s const [] | ].
    split; [ auto | intros s H; right; exact H ].
  - apply andb_true_iff in Hcodes as [Hid Hrest].
    apply negb_true_iff in Hid.
    assert (Hnew : lookup id (static_code statics) = None).
    { apply Hfresh_code. cbn [flat_map code_ids_of_binding app].
      rewrite among_cons, Pos.eqb_refl. reflexivity. }
    rewrite Hnew.
    destruct
      (IH (Static_env ((id, c) :: static_code statics) (static_values statics))
         Hhoisted Hrest Hsymbols)
      as (bound_code & bound_values & Heval & Hbound & Hold & Hconverse
          & Hbound_s & Hold_s & Hconverse_s).
    { intros id' Hid'. simpl.
      destruct (Pos.eqb_spec id' id) as [-> | _];
        [ rewrite Hid in Hid'; discriminate | ].
      apply Hfresh_code. cbn [flat_map code_ids_of_binding app].
      rewrite among_cons, Hid'. apply orb_true_r. }
    { exact Hfresh_symbols. }
    exists bound_code, bound_values. split; [ exact Heval | ].
    split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
    + intros id' c' [Heq | Hin]; [ | exact (Hbound id' c' Hin) ].
      injection Heq as <- <-. apply Hold. simpl. rewrite Pos.eqb_refl.
      reflexivity.
    + intros id' c' Hlookup. apply Hold. simpl.
      destruct (Pos.eqb_spec id' id) as [-> | _]; [ | exact Hlookup ].
      rewrite Hnew in Hlookup. discriminate.
    + intros id' Hid'. cbn [flat_map code_ids_of_binding app].
      try rewrite among_cons.
      destruct (Hconverse id' Hid') as [Hamong | Hlookup].
      * left.
        first
          [ rewrite Hamong; apply orb_true_r
          | apply among_In; right; apply among_In; exact Hamong ].
      * simpl in Hlookup. destruct (Pos.eqb_spec id' id) as [-> | _].
        -- left.
           first
             [ reflexivity
             | rewrite Pos.eqb_refl; reflexivity
             | apply among_In; left; reflexivity ].
        -- right. exact Hlookup.
    + intros s const [Heq | Hin]; [ discriminate | ].
      exact (Hbound_s s const Hin).
    + exact Hold_s.
    + exact Hconverse_s.
  - destruct (eval_static_const [] const) as [allocated | ] eqn:Hconst;
      [ | discriminate ].
    simpl in Hhoisted.
    apply andb_true_iff in Hsymbols as [Hsym Hrest].
    apply negb_true_iff in Hsym.
    assert (Hnew : lookup sym (static_values statics) = None).
    { apply Hfresh_symbols. cbn [flat_map symbols_of_binding app].
      rewrite among_cons, Pos.eqb_refl. reflexivity. }
    rewrite Hnew. simpl.
    destruct
      (IH (Static_env (static_code statics)
             ((sym, allocated) :: static_values statics))
         Hhoisted Hcodes Hrest)
      as (bound_code & bound_values & Heval & Hbound & Hold & Hconverse
          & Hbound_s & Hold_s & Hconverse_s).
    { exact Hfresh_code. }
    { intros s' Hs'. simpl.
      destruct (Pos.eqb_spec s' sym) as [-> | _];
        [ rewrite Hsym in Hs'; discriminate | ].
      apply Hfresh_symbols. cbn [flat_map symbols_of_binding app].
      rewrite among_cons, Hs'. apply orb_true_r. }
    exists bound_code, bound_values. split; [ exact Heval | ].
    split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
    + intros id' c' [Heq | Hin]; [ discriminate | ].
      exact (Hbound id' c' Hin).
    + exact Hold.
    + exact Hconverse.
    + intros s const' [Heq | Hin]; [ | exact (Hbound_s s const' Hin) ].
      injection Heq as <- <-. exists allocated.
      split; [ exact Hconst | ]. apply Hold_s. simpl.
      rewrite Pos.eqb_refl. reflexivity.
    + intros s allocated' Hlookup. apply Hold_s. simpl.
      destruct (Pos.eqb_spec s sym) as [-> | _]; [ | exact Hlookup ].
      rewrite Hnew in Hlookup. discriminate.
    + intros s Hs. cbn [flat_map symbols_of_binding app].
      try rewrite among_cons.
      destruct (Hconverse_s s Hs) as [Hamong | Hlookup].
      * left.
        first
          [ rewrite Hamong; apply orb_true_r
          | apply among_In; right; apply among_In; exact Hamong ].
      * simpl in Hlookup. destruct (Pos.eqb_spec s sym) as [-> | _].
        -- left.
           first
             [ reflexivity
             | rewrite Pos.eqb_refl; reflexivity
             | apply among_In; left; reflexivity ].
        -- right. exact Hlookup.
Qed.

(* ======================================================================== *)
(* The bindings of the groups of a unit                                     *)
(* ======================================================================== *)

(* All the bindings of the groups of static bindings of an expression,
   in the order in which [hoisted_bindings] takes those that it takes. *)
Fixpoint toplevel_bindings (e : expr) : list static_binding :=
  match e with
  | Let _ _ _ body => toplevel_bindings body
  | Let_static bindings body => bindings ++ toplevel_bindings body
  | Let_cont_non_recursive _ h body =>
    toplevel_bindings body ++ handler_toplevel_bindings h
  | Let_cont_recursive _ body => toplevel_bindings body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_toplevel_bindings (h : continuation_handler)
    : list static_binding :=
  match h with
  | Handler _ _ handler => toplevel_bindings handler
  end.

Fixpoint hoisted_bindings_filter (e : expr) {struct e} :
  hoisted_bindings e = filter is_hoisted (toplevel_bindings e)
with handler_hoisted_bindings_filter (h : continuation_handler) {struct h} :
  handler_hoisted_bindings h
  = filter is_hoisted (handler_toplevel_bindings h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [hoisted_bindings toplevel_bindings]; try reflexivity.
    + apply hoisted_bindings_filter.
    + rewrite filter_app, (hoisted_bindings_filter body). reflexivity.
    + rewrite filter_app, (hoisted_bindings_filter body),
        (handler_hoisted_bindings_filter h).
      reflexivity.
    + apply hoisted_bindings_filter.
  - destruct h as [params metadata handler].
    cbn [handler_hoisted_bindings handler_toplevel_bindings].
    apply hoisted_bindings_filter.
Qed.

(* Their names are among the static names of the expression, and are
   distinct where those are. *)
Fixpoint toplevel_names_in (names_of : static_binding -> list positive)
    (e : expr) {struct e} :
  incl (flat_map names_of (toplevel_bindings e)) (static_names names_of e)
with handler_toplevel_names_in (names_of : static_binding -> list positive)
    (h : continuation_handler) {struct h} :
  incl (flat_map names_of (handler_toplevel_bindings h))
    (handler_static_names names_of h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [toplevel_bindings static_names]; try (intros n []).
    + apply toplevel_names_in.
    + rewrite flat_map_app.
      apply incl_app_app; [ apply incl_refl | apply toplevel_names_in ].
    + rewrite flat_map_app.
      apply incl_app;
        [ apply incl_appr, toplevel_names_in
        | apply incl_appl, handler_toplevel_names_in ].
    + apply incl_appr, toplevel_names_in.
  - destruct h as [params metadata handler].
    cbn [handler_toplevel_bindings handler_static_names].
    apply toplevel_names_in.
Qed.

Fixpoint toplevel_names_nodup (names_of : static_binding -> list positive)
    (e : expr) {struct e} :
  NoDup (static_names names_of e) ->
  NoDup (flat_map names_of (toplevel_bindings e))
with handler_toplevel_names_nodup
    (names_of : static_binding -> list positive) (h : continuation_handler)
    {struct h} :
  NoDup (handler_static_names names_of h) ->
  NoDup (flat_map names_of (handler_toplevel_bindings h)).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [toplevel_bindings static_names];
      try (intros _; apply NoDup_nil).
    + apply toplevel_names_nodup.
    + intros H. rewrite flat_map_app. apply NoDup_app.
      * exact (nodup_app_l _ _ H).
      * exact (toplevel_names_nodup names_of body (nodup_app_r _ _ H)).
      * intros n Hn Hn'.
        exact (nodup_app_disjoint _ _ H n Hn
                 (toplevel_names_in names_of body n Hn')).
    + intros H. rewrite flat_map_app. apply NoDup_app.
      * exact (toplevel_names_nodup names_of body (nodup_app_r _ _ H)).
      * exact (handler_toplevel_names_nodup names_of h (nodup_app_l _ _ H)).
      * intros n Hn Hn'.
        exact (nodup_app_disjoint _ _ H n
                 (handler_toplevel_names_in names_of h n Hn')
                 (toplevel_names_in names_of body n Hn)).
    + intros H.
      exact (toplevel_names_nodup names_of body (nodup_app_r _ _ H)).
  - destruct h as [params metadata handler].
    cbn [handler_toplevel_bindings handler_static_names].
    apply toplevel_names_nodup.
Qed.

(* The names of the bindings that a filter keeps. *)
Lemma in_flat_map_filter {A B : Type} (f : A -> list B) (keep : A -> bool)
    (l : list A) (b : B) :
  b ∈ flat_map f (filter keep l) -> b ∈ flat_map f l.
Proof.
  intros H. apply in_flat_map in H as (a & Ha & Hb). apply in_flat_map.
  exists a. split; [ exact (proj1 (proj1 (filter_In _ _ _) Ha)) | exact Hb ].
Qed.

Lemma nodup_flat_map_filter {A B : Type} (f : A -> list B) (keep : A -> bool)
    (l : list A) :
  NoDup (flat_map f l) -> NoDup (flat_map f (filter keep l)).
Proof.
  induction l as [ | a l IH]; cbn [flat_map filter]; intros H;
    [ apply NoDup_nil | ].
  destruct (keep a); cbn [flat_map].
  - apply NoDup_app.
    + exact (nodup_app_l _ _ H).
    + exact (IH (nodup_app_r _ _ H)).
    + intros b Hb Hb'.
      exact (nodup_app_disjoint _ _ H b Hb
               (in_flat_map_filter f keep l b Hb')).
  - exact (IH (nodup_app_r _ _ H)).
Qed.

(* Where no symbol is bound twice, the symbol of a binding that is not
   taken out is not that of one that is. *)
Lemma kept_symbol_not_hoisted (s : symbol) (const : static_const)
    (bindings : list static_binding) :
  is_hoisted (Bind_symbol s const) = false ->
  NoDup (flat_map symbols_of_binding bindings) ->
  Bind_symbol s const ∈ bindings ->
  ¬ s ∈ flat_map symbols_of_binding (filter is_hoisted bindings).
Proof.
  intros Hkept.
  induction bindings as [ | binding bindings IH]; cbn [flat_map filter];
    intros Hnodup Hin; [ destruct Hin | ].
  destruct Hin as [-> | Hin].
  - rewrite Hkept. intros Hs.
    apply (nodup_app_disjoint _ _ Hnodup s);
      [ left; reflexivity | exact (in_flat_map_filter _ _ _ _ Hs) ].
  - assert (Hs : s ∈ flat_map symbols_of_binding bindings).
    { apply in_flat_map. exists (Bind_symbol s const).
      split; [ exact Hin | left; reflexivity ]. }
    destruct (is_hoisted binding); cbn [flat_map].
    + intros Hs'. apply in_app_or in Hs' as [Hfirst | Hrest].
      * exact (nodup_app_disjoint _ _ Hnodup s Hfirst Hs).
      * exact (IH (nodup_app_r _ _ Hnodup) Hin Hrest).
    + exact (IH (nodup_app_r _ _ Hnodup) Hin).
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Section Hoisted.

(* The code ids that the unit declares deleted, and what the unit with
   its closed bindings first has bound once its first group is
   evaluated. *)
Variable deleted : list code_id.
Variable src_code : list (code_id * code).
Variable src_values : list (symbol * statically_allocated_value).

(* A code id that the unit declares deleted is not bound first. *)
Hypothesis deleted_not_src :
  ∀ id : code_id, among id deleted = true -> lookup id src_code = None.

(* A symbol that is bound first, and one that is not. *)
Definition hoisted_symbol (s : symbol) : bool := is_bound s src_values.

Definition kept_symbol (s : symbol) : bool := negb (hoisted_symbol s).

(* The second state: the first with the bindings that are bound first
   taken out of its terms; all the code bound; and the symbols bound
   first beside those that the first state has bound and that are not
   among them. *)
Fixpoint hoist_cont (c : cont_value) : cont_value :=
  match c with
  | Closure h vars conts recursive =>
    Closure (handler_without_hoisted h) vars
      (map (fun binding => (fst binding, hoist_cont (snd binding))) conts)
      recursive
  | Outer k => Outer k
  | Extra_args c extra => Extra_args (hoist_cont c) extra
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    Over_application remaining remaining_arity return_arity
      (hoist_cont return_to) (hoist_cont exn_to)
  end.

Definition hoist_conts (conts : cont_env) : cont_env :=
  map (fun binding => (fst binding, hoist_cont (snd binding))) conts.

Definition keep_values
    (values : list (symbol * statically_allocated_value))
    : list (symbol * statically_allocated_value) :=
  filter (fun binding => kept_symbol (fst binding)) values.

Definition hoist_env (statics : static_env) : static_env :=
  Static_env src_code (keep_values (static_values statics) ++ src_values).

Definition hoist_state (s : state) : state :=
  State (without_hoisted (state_expr s)) (state_vars s)
    (hoist_conts (state_conts s)) (hoist_env (state_statics s)) (state_heap s)
      (state_reads s).

(* What a symbol is bound to in the second static environment. *)
Lemma lookup_hoist_env (statics : static_env) (s : symbol) :
  lookup s (static_values (hoist_env statics))
  = if hoisted_symbol s then lookup s src_values
    else lookup s (static_values statics).
Proof.
  cbn [hoist_env static_values]. rewrite lookup_app. unfold keep_values.
  destruct (hoisted_symbol s) eqn:Hs.
  - rewrite (lookup_filter_dropped kept_symbol s);
      [ reflexivity | unfold kept_symbol; rewrite Hs; reflexivity ].
  - rewrite (lookup_filter_kept kept_symbol s);
      [ | unfold kept_symbol; rewrite Hs; reflexivity ].
    destruct (lookup s (static_values statics)); [ reflexivity | ].
    unfold hoisted_symbol, is_bound in Hs.
    destruct (lookup s src_values); [ discriminate | reflexivity ].
Qed.

Lemma resolve_hoist (k : continuation) (conts : cont_env) :
  resolve k (hoist_conts conts) = hoist_cont (resolve k conts).
Proof.
  unfold resolve, hoist_conts. rewrite lookup_map_snd.
  destruct (lookup k conts); reflexivity.
Qed.

Lemma with_extra_args_hoist (c : cont_value) (extra : list value) :
  hoist_cont (with_extra_args c extra) = with_extra_args (hoist_cont c) extra.
Proof. destruct extra; reflexivity. Qed.

Lemma rec_closures_hoist
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  handlers_bind_statics handlers = false ->
  rec_closures handlers vars (hoist_conts conts)
  = hoist_conts (rec_closures handlers vars conts).
Proof.
  intros Hnone. unfold rec_closures, hoist_conts. rewrite map_map.
  apply map_ext_in. intros [k h] Hin. cbn [fst snd hoist_cont].
  rewrite (proj1 (handlers_without_hoisted_nothing handlers Hnone k h Hin)).
  reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* What the terms still bind                                                *)
(* ------------------------------------------------------------------------ *)

(* A binding of the terms of the first state, for the second: code is
   bound; a symbol that is taken out is bound to the value of its
   constant; one that is not is not a symbol bound first. *)
Definition binding_known (binding : static_binding) : Prop :=
  match binding with
  | Bind_code id c => lookup id src_code = Some c
  | Bind_deleted_code _ => True
  | Bind_symbol s const =>
    match eval_static_const [] const with
    | Some allocated => lookup s src_values = Some allocated
    | None => lookup s src_values = None
    end
  end.

Fixpoint expr_known (e : expr) : Prop :=
  match e with
  | Let _ _ _ body => expr_known body
  | Let_static bindings body =>
    Forall binding_known bindings ∧ expr_known body
  | Let_cont_non_recursive _ h body => handler_known h ∧ expr_known body
  | Let_cont_recursive _ body => expr_known body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => True
  end
with handler_known (h : continuation_handler) : Prop :=
  match h with
  | Handler _ _ handler => expr_known handler
  end.

(* An expression that binds nothing static. *)
Fixpoint expr_known_nothing (e : expr) {struct e} :
  binds_statics e = false -> expr_known e
with handler_known_nothing (h : continuation_handler) {struct h} :
  handler_binds_statics h = false -> handler_known h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [binds_statics expr_known]; try (intros _; exact I).
    + apply expr_known_nothing.
    + intros H. apply orb_false_iff in H as [Hbindings Hbody].
      split; [ | exact (expr_known_nothing body Hbody) ].
      clear Hbody. induction bindings as [ | binding bindings IH];
        [ apply Forall_nil | ].
      cbn [existsb] in Hbindings.
      apply orb_false_iff in Hbindings as [Hbinding Hrest].
      apply Forall_cons; [ | exact (IH Hrest) ].
      destruct binding as [id c | id | sym const]; simpl in Hbinding;
        try discriminate.
      exact I.
    + intros H. apply orb_false_iff in H as [Hh Hbody].
      split;
        [ exact (handler_known_nothing h Hh)
        | exact (expr_known_nothing body Hbody) ].
    + intros H. apply orb_false_iff in H as [_ Hbody].
      exact (expr_known_nothing body Hbody).
  - destruct h as [params metadata handler].
    cbn [handler_binds_statics handler_known].
    apply expr_known_nothing.
Qed.

(* An expression all of whose bindings are known. *)
Fixpoint expr_known_of_bindings (e : expr) {struct e} :
  Forall binding_known (toplevel_bindings e) -> expr_known e
with handler_known_of_bindings (h : continuation_handler) {struct h} :
  Forall binding_known (handler_toplevel_bindings h) -> handler_known h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [toplevel_bindings expr_known]; try (intros _; exact I).
    + apply expr_known_of_bindings.
    + intros H. apply Forall_app in H as [Hbindings Hbody].
      exact (conj Hbindings (expr_known_of_bindings body Hbody)).
    + intros H. apply Forall_app in H as [Hbody Hh].
      exact (conj (handler_known_of_bindings h Hh)
               (expr_known_of_bindings body Hbody)).
    + apply expr_known_of_bindings.
  - destruct h as [params metadata handler].
    cbn [handler_toplevel_bindings handler_known].
    apply expr_known_of_bindings.
Qed.

Fixpoint cont_known (c : cont_value) : Prop :=
  match c with
  | Outer _ => True
  | Closure h _ conts recursive =>
    handler_known h
    ∧ handlers_bind_statics recursive = false
    ∧ fold_right (fun kc acc => cont_known (snd kc) ∧ acc) True conts
  | Extra_args c _ => cont_known c
  | Over_application _ _ _ return_to exn_to =>
    cont_known return_to ∧ cont_known exn_to
  end.

Definition conts_known (conts : cont_env) : Prop :=
  fold_right (fun kc acc => cont_known (snd kc) ∧ acc) True conts.

Lemma conts_known_app (conts conts' : cont_env) :
  conts_known conts -> conts_known conts' -> conts_known (conts ++ conts').
Proof.
  unfold conts_known.
  induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma resolve_known (k : continuation) (conts : cont_env) :
  conts_known conts -> cont_known (resolve k conts).
Proof.
  unfold resolve, conts_known.
  induction conts as [ | [k' c] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma with_extra_args_known (c : cont_value) (extra : list value) :
  cont_known c -> cont_known (with_extra_args c extra).
Proof. destruct extra; intros Hc; exact Hc. Qed.

Lemma rec_closures_known (vars : env) (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics all = false ->
  conts_known conts ->
  (∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers -> (k, h) ∈ all) ->
  conts_known (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros Hall Hconts. unfold conts_known.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; exact I | ].
  intros Hin. split; [ | apply IH; intros k' h' H; apply Hin; right; exact H ].
  split; [ | split; [ exact Hall | exact Hconts ] ].
  apply handler_known_nothing.
  unfold handlers_bind_statics in Hall.
  assert (Hmember : (k, h) ∈ all) by (apply Hin; left; reflexivity).
  clear Hin IH. induction all as [ | [k' h'] all IHall]; [ destruct Hmember | ].
  simpl in Hall. apply orb_false_iff in Hall as [Hh Hrest].
  destruct Hmember as [Heq | Hmember]; [ | exact (IHall Hrest Hmember) ].
  injection Heq as _ <-. exact Hh.
Qed.

(* What the first static environment has bound, the second has bound to
   the same: the code, and the symbols that are bound first. *)
Definition statics_known (statics : static_env) : Prop :=
  (∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c -> lookup id src_code = Some c)
  ∧ (∀ (sym : symbol) (allocated : statically_allocated_value),
      lookup sym (static_values statics) = Some allocated ->
      hoisted_symbol sym = true ->
      lookup sym src_values = Some allocated).

Definition known_inv (s : state) : Prop :=
  expr_known (state_expr s)
  ∧ conts_known (state_conts s)
  ∧ statics_known (state_statics s).

Definition wf_inv : state -> Prop :=
  inv code_ids_of_binding code_id_bound.

(* The code that is bound binds nothing static. *)
Lemma wf_inv_code (s : state) (id : code_id) (c : code) :
  wf_inv s ->
  lookup id (static_code (state_statics s)) = Some c ->
  binds_statics (code_body c) = false.
Proof.
  intros (_ & _ & _ & _ & _ & S) Hlookup. pose proof (S id c Hlookup) as Hc.
  destruct c as [params my_closure ret exn metadata body]. simpl in Hc |- *.
  exact (proj1 (proj2 (statics_nowhere code_ids_of_binding body Hc))).
Qed.

Lemma known_inv_over (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_known
    (Over_application remaining remaining_arity return_arity return_to
       exn_to)
  <-> cont_known return_to ∧ cont_known exn_to.
Proof. reflexivity. Qed.

Lemma known_inv_code (statics : static_env) (hp : heap) (rd : read_record) :
  statics_known statics ->
  (∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c ->
    binds_statics (code_body c) = false) ->
  ∀ (called : code_id) (c : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some c ->
    cont_known return_to ->
    cont_known exn_to ->
    known_inv
      (State (code_body c) callee_vars
         [(code_return_continuation c, return_to);
          (code_exn_continuation c, exn_to)]
         statics hp rd).
Proof.
  intros S Hnone called c callee_vars return_to exn_to Hlookup Hreturn Hexn.
  unfold known_inv. simpl_state.
  refine (conj _ (conj (conj Hreturn (conj Hexn I)) S)).
  exact (expr_known_nothing _ (Hnone called c Hlookup)).
Qed.

Lemma known_inv_handler (statics : static_env) (hp : heap) (rd : read_record) :
  statics_known statics ->
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env),
    cont_known
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    known_inv
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros S params metadata handler def_vars def_conts recursive handler_vars
    Cr.
  unfold known_inv. simpl_state. simpl in Cr.
  destruct Cr as (Hh & Hrec & Cdef).
  refine (conj Hh (conj (conts_known_app _ _ _ Cdef) S)).
  exact (rec_closures_known def_vars def_conts recursive recursive Hrec Cdef
           (fun k h H => H)).
Qed.

Lemma known_inv_step (s s' : state) :
  wf_inv s -> known_inv s -> step s = Step s' -> known_inv s'.
Proof.
  intros W (K & C & S) Hstep.
  assert (Scode :
    ∀ (id : code_id) (c : code),
      lookup id (static_code (state_statics s)) = Some c ->
      binds_statics (code_body c) = false)
    by (intros id c H; exact (wf_inv_code s id c W H)).
  clear W. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep; cbn [expr_known] in K.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. exact (conj K (conj C S)).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. destruct K as [Kbindings Kbody].
    destruct S as [Scodes Svalues]. unfold known_inv. simpl_state.
    refine (conj Kbody (conj C (conj _ _))).
    + intros id c Hlookup.
      destruct (FlambdaHoistCodeProofs.eval_static_bindings_code_in vars
                  bindings statics statics' id c Heval Hlookup)
        as [Hold | Hin]; [ exact (Scodes id c Hold) | ].
      exact (proj1 (Forall_forall _ _) Kbindings _ Hin).
    + intros sym allocated Hlookup Hsym.
      destruct (FlambdaClosuresProofs.eval_static_bindings_symbol vars
                  bindings statics statics' sym allocated Heval Hlookup)
        as [Hold | (const & Hin & Hconst)];
        [ exact (Svalues sym allocated Hold Hsym) | ].
      pose proof (proj1 (Forall_forall _ _) Kbindings _ Hin) as Hknown.
      cbn [binding_known] in Hknown.
      destruct (eval_static_const [] const) as [closed | ] eqn:Hclosed.
      * rewrite (eval_static_const_closed vars const closed Hclosed)
          in Hconst.
        injection Hconst as <-. exact Hknown.
      * unfold hoisted_symbol, is_bound in Hsym. rewrite Hknown in Hsym.
        discriminate.
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. destruct K as [Kh Kbody].
    unfold known_inv. simpl_state.
    refine (conj Kbody (conj (conj _ C) S)).
    simpl. exact (conj Kh (conj eq_refl C)).
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ discriminate | ].
    injection Hstep as <-. unfold known_inv. simpl_state.
    refine (conj K (conj (conts_known_app _ _ _ C) S)).
    exact (rec_closures_known vars conts handlers handlers Hbinds C
             (fun k h H => H)).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands statics _ rd_ cont_known known_inv
         (fun c extra H => H)
         known_inv_over (known_inv_code statics _ rd_ S Scode)
         (known_inv_handler statics _ rd_ S) _ _ _ _ _ _ _ s'
         (resolve_known _ conts C) _ Hstep).
    apply with_extra_args_known, resolve_known, C.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands statics _ rd_ cont_known known_inv
         (fun c extra H => H) known_inv_over
         (known_inv_code statics _ rd_ S Scode)
         (known_inv_handler statics _ rd_ S) _ vs s' (resolve_known _ conts C)
         Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    injection Hstep as <-. exact (conj I (conj C S)).
  - discriminate.
Qed.

(* The invariant of the first run: no static name is bound twice; every
   static name that the state mentions or holds is in scope and bound;
   and what it still binds, the second run has bound. *)
Definition hoist_inv (s : state) : Prop :=
  wf_inv s
  ∧ inv symbols_of_binding symbol_bound s
  ∧ terms_inv deleted s
  ∧ state_statics_bound deleted s
  ∧ known_inv s.

Lemma hoist_inv_step (s s' : state) :
  hoist_inv s -> step s = Step s' -> hoist_inv s'.
Proof.
  intros (W & Ws & T & V & K) Hstep.
  pose proof (values_step deleted s T V) as V'.
  rewrite Hstep in V'.
  exact (conj (inv_step code_ids_of_binding code_id_bound
                 code_bound_after s s' W Hstep)
           (conj (inv_step symbols_of_binding symbol_bound
                    symbol_bound_after s s' Ws Hstep)
              (conj (terms_inv_step deleted s s' T Hstep)
                 (conj V' (known_inv_step s s' W K Hstep))))).
Qed.

(* ------------------------------------------------------------------------ *)
(* The two runs in step                                                     *)
(* ------------------------------------------------------------------------ *)

(* What the second run does: the same step, to the second state of the
   result, the same exit, or nothing where the first does nothing. *)
Definition result_hoisted (result result' : step_result) : Prop :=
  match result with
  | Step t => result' = Step (hoist_state t)
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Undefined => result' = Undefined
  end.

(* A group, with its bindings that are bound first taken out, binds in
   the second run what it binds in the first of the names that are not
   bound first, where the first does not stop on a name that is bound
   already. *)
Lemma eval_static_bindings_hoist (vars : env)
    (bindings : list static_binding) :
  ∀ statics : static_env,
    Forall binding_known bindings ->
    rebinds code_ids_of_binding (code_id_bound statics) bindings = false ->
    rebinds symbols_of_binding (symbol_bound statics) bindings = false ->
    eval_static_bindings vars
      (filter (fun binding => negb (is_hoisted binding)) bindings)
      (hoist_env statics)
    = option_map hoist_env (eval_static_bindings vars bindings statics).
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics Hknown Hcodes Hsymbols; [ reflexivity | | | ];
    apply Forall_cons_iff in Hknown as [Hbinding Hrest];
    cbn [filter is_hoisted negb eval_static_bindings].
  - unfold rebinds in Hcodes. simpl in Hcodes.
    apply orb_false_iff in Hcodes as [Hdistinct Hbound].
    apply negb_false_iff, andb_true_iff in Hdistinct as [Hfresh Hdistinct].
    apply negb_true_iff in Hfresh.
    apply orb_false_iff in Hbound as [Hid Hbound].
    unfold code_id_bound in Hid.
    destruct (lookup id (static_code statics)); [ discriminate | ].
    rewrite <-
      (IH (Static_env ((id, c) :: static_code statics)
             (static_values statics))
         Hrest);
      [ reflexivity | | exact Hsymbols ].
    unfold rebinds. rewrite Hdistinct. cbn [negb orb].
    apply not_true_is_false. intros Hexists.
    apply existsb_exists in Hexists as [n [Hin Hn]].
    unfold code_id_bound in Hn. simpl in Hn.
    destruct (Pos.eqb_spec n id) as [-> | _].
    + apply among_In in Hin. rewrite Hfresh in Hin. discriminate.
    + assert (Hold :
        existsb (code_id_bound statics)
          (flat_map code_ids_of_binding bindings)
        = true)
        by (apply existsb_exists; exists n; split; assumption).
      exact (diff_true_false (eq_trans (eq_sym Hold) Hbound)).
  - exact (IH statics Hrest Hcodes Hsymbols).
  - unfold rebinds in Hsymbols. simpl in Hsymbols.
    apply orb_false_iff in Hsymbols as [Hdistinct Hbound].
    apply negb_false_iff, andb_true_iff in Hdistinct as [Hfresh Hdistinct].
    apply negb_true_iff in Hfresh.
    apply orb_false_iff in Hbound as [Hsym Hbound].
    unfold symbol_bound in Hsym.
    destruct (lookup sym (static_values statics)) eqn:Hlookup;
      [ discriminate | ].
    assert (Hnext :
      ∀ allocated : statically_allocated_value,
        rebinds symbols_of_binding
          (symbol_bound
             (Static_env (static_code statics)
                ((sym, allocated) :: static_values statics)))
          bindings
        = false).
    { intros allocated. unfold rebinds. rewrite Hdistinct. cbn [negb orb].
      apply not_true_is_false. intros Hexists.
      apply existsb_exists in Hexists as [n [Hin Hn]].
      unfold symbol_bound in Hn. simpl in Hn.
      destruct (Pos.eqb_spec n sym) as [-> | _].
      - apply among_In in Hin. rewrite Hfresh in Hin. discriminate.
      - assert (Hold :
          existsb (symbol_bound statics)
            (flat_map symbols_of_binding bindings)
          = true)
          by (apply existsb_exists; exists n; split; assumption).
        exact (diff_true_false (eq_trans (eq_sym Hold) Hbound)). }
    cbn [binding_known] in Hbinding.
    destruct (eval_static_const [] const) as [closed | ] eqn:Hclosed;
      cbn [negb filter eval_static_bindings].
    + (* Taken out: the second run has it bound. *)
      rewrite (eval_static_const_closed vars const closed Hclosed).
      change
        (eval_static_bindings vars
           (filter (fun binding => negb (is_hoisted binding)) bindings)
           (hoist_env statics)
         = option_map hoist_env
             (eval_static_bindings vars bindings
                (Static_env (static_code statics)
                   ((sym, closed) :: static_values statics)))).
      rewrite <-
        (IH (Static_env (static_code statics)
               ((sym, closed) :: static_values statics))
           Hrest Hcodes (Hnext closed)).
      f_equal.
      unfold hoist_env, keep_values. cbn [static_values filter fst].
      unfold kept_symbol, hoisted_symbol, is_bound. rewrite Hbinding.
      reflexivity.
    + (* Kept: bound in both, and not a symbol that is bound first. *)
      assert (Hkept : hoisted_symbol sym = false)
        by (unfold hoisted_symbol, is_bound; rewrite Hbinding; reflexivity).
      rewrite lookup_hoist_env, Hkept, Hlookup.
      destruct (eval_static_const vars const) as [allocated | ];
        [ | reflexivity ].
      change
        (eval_static_bindings vars
           (filter (fun binding => negb (is_hoisted binding)) bindings)
           (Static_env (static_code (hoist_env statics))
              ((sym, allocated) :: static_values (hoist_env statics)))
         = option_map hoist_env
             (eval_static_bindings vars bindings
                (Static_env (static_code statics)
                   ((sym, allocated) :: static_values statics)))).
      rewrite <-
        (IH (Static_env (static_code statics)
               ((sym, allocated) :: static_values statics))
           Hrest Hcodes (Hnext allocated)).
      f_equal.
      unfold hoist_env, keep_values. cbn [static_values static_code filter fst].
      unfold kept_symbol. rewrite Hkept. reflexivity.
Qed.

(* A full call. *)
Lemma call_code_hoist (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  result_hoisted
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity (hoist_cont return_to)
       (hoist_cont exn_to) (hoist_env statics) hp rd).
Proof.
  unfold result_hoisted, call_code.
  destruct (binds_statics (code_body c)) eqn:Hbinds; [ reflexivity | ].
  match goal with
  | |- context [negb ?agrees] => destruct (negb agrees)
  end;
    [ reflexivity | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)]);
    [ | reflexivity ].
  unfold hoist_state. simpl_state.
  rewrite (proj1 (without_hoisted_nothing _ Hbinds)). reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* What the two static environments agree on                                *)
(* ------------------------------------------------------------------------ *)

(* A symbol that the first static environment has bound is bound to the
   same in the second. *)
Lemma lookup_bound_hoist (statics : static_env) (s : symbol) :
  statics_known statics ->
  among s (bound_symbols statics) = true ->
  lookup s (static_values (hoist_env statics))
  = lookup s (static_values statics).
Proof.
  intros [_ Svalues] Hbound. rewrite lookup_hoist_env.
  destruct (hoisted_symbol s) eqn:Hs; [ | reflexivity ].
  unfold bound_symbols in Hbound. rewrite <- is_bound_among in Hbound.
  unfold is_bound in Hbound.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
    [ | discriminate ].
  exact (Svalues s allocated Hlookup Hs).
Qed.

(* So a value that mentions only what is bound is read alike. *)
Lemma symbol_bound_hoist (statics : static_env) (v : value) :
  statics_known statics ->
  value_statics_bound deleted statics v ->
  ∀ s : symbol,
    v = Value_symbol s ->
    lookup s (static_values (hoist_env statics))
    = lookup s (static_values statics).
Proof.
  intros S Hv s ->. apply (lookup_bound_hoist statics s S). exact Hv.
Qed.

Lemma closure_of_hoist (statics : static_env) (v : value) :
  statics_known statics ->
  value_statics_bound deleted statics v ->
  closure_of (hoist_env statics) v = closure_of statics v.
Proof.
  intros S Hv.
  exact (closure_of_static_values (hoist_env statics) statics v
           (symbol_bound_hoist statics v S Hv)).
Qed.

Lemma closure_in_set_hoist (statics : static_env) (v : value) :
  statics_known statics ->
  value_statics_bound deleted statics v ->
  closure_in_set (hoist_env statics) v = closure_in_set statics v.
Proof.
  intros S Hv. pose proof (symbol_bound_hoist statics v S Hv) as Hsame.
  destruct v; unfold closure_in_set; try reflexivity.
  rewrite (Hsame _ eq_refl). reflexivity.
Qed.

Lemma project_function_slot_hoist (statics : static_env)
    (move_from move_to : function_slot) (v : value) :
  statics_known statics ->
  value_statics_bound deleted statics v ->
  project_function_slot (hoist_env statics) move_from move_to v
  = project_function_slot statics move_from move_to v.
Proof.
  intros S Hv. pose proof (symbol_bound_hoist statics v S Hv) as Hsame.
  destruct v; unfold project_function_slot; try reflexivity.
  rewrite (Hsame _ eq_refl). reflexivity.
Qed.

(* The value of a defining expression that mentions only what is in
   scope, under variables whose values mention only what is bound. *)
Lemma eval_named_hoist (statics : static_env) (vars : env) (d : named) :
  statics_known statics ->
  env_values_all (value_statics_bound deleted statics) vars ->
  named_statics_in_scope (bound_symbols statics)
    (bound_code_ids statics ++ deleted) d
  = true ->
  eval_named (hoist_env statics) vars d = eval_named statics vars d.
Proof.
  intros S Hvars Hscope.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    cbn [eval_named named_statics_in_scope] in Hscope |- *;
    try reflexivity.
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | reflexivity ].
    simpl. apply eval_prim_static_values. intros s Hin.
    apply (lookup_bound_hoist statics s S).
    exact (proj1 (forallb_forall _ _)
             (eval_simples_among _ _ vars args Hvars vs Hscope Hargs)
             (Value_symbol s) Hin).
  - destruct (eval_simple vars closure) as [v | ] eqn:Hclosure;
      [ | reflexivity ].
    simpl.
    rewrite (closure_of_hoist statics v S
               (eval_simple_among _ _ vars closure v Hvars Hscope Hclosure)).
    reflexivity.
  - destruct (eval_simple vars closure) as [v | ] eqn:Hclosure;
      [ | reflexivity ].
    simpl. unfold project_value_slot.
    rewrite (closure_in_set_hoist statics v S
               (eval_simple_among _ _ vars closure v Hvars Hscope Hclosure)).
    reflexivity.
  - destruct (eval_simple vars closure) as [v | ] eqn:Hclosure;
      [ | reflexivity ].
    simpl.
    exact (project_function_slot_hoist statics move_from move_to v S
             (eval_simple_among _ _ vars closure v Hvars Hscope Hclosure)).
Qed.

(* And it does the same on a heap. *)
Lemma exec_named_hoist (statics : static_env) (hp : heap) (rd : read_record)
    (vars : env)
    (d : named) :
  statics_known statics ->
  env_values_all (value_statics_bound deleted statics) vars ->
  named_statics_in_scope (bound_symbols statics)
    (bound_code_ids statics ++ deleted) d
  = true ->
  exec_named (hoist_env statics) hp rd vars d = exec_named statics hp rd vars d.
Proof.
  intros S Hvars Hscope. apply exec_named_same.
  - exact (eval_named_hoist statics vars d S Hvars Hscope).
  - intros p args ->. split; [ reflexivity | ].
    intros vs s Hargs Hin. cbn [named_statics_in_scope] in Hscope.
    apply (lookup_bound_hoist statics s S).
    exact (proj1 (forallb_forall _ _)
             (eval_simples_among _ _ vars args Hvars vs Hscope Hargs)
             (Value_symbol s) Hin).
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma value_bound_partial (statics : static_env) (callee : value)
    (args : list value) (args_arity : arity) :
  value_statics_bound deleted statics
    (Value_partial_application callee args args_arity)
  <-> value_statics_bound deleted statics callee
      ∧ Forall (value_statics_bound deleted statics) args.
Proof.
  unfold value_statics_bound. simpl.
  rewrite andb_true_iff, forallb_Forall_among. reflexivity.
Qed.

(* What the first static environment binds its symbols to mentions only
   what it binds. *)
Definition statics_fine (statics : static_env) : Prop :=
  Forall
    (fun binding => allocated_statics_bound deleted statics (snd binding))
    (static_values statics).

(* The fields of a tuple that mentions only what is bound: of a block,
   or of a symbol bound to one. *)
Lemma tuple_fields_bound (statics : static_env) :
  statics_fine statics ->
  P_fields (value_statics_bound deleted statics) statics.
Proof.
  intros Hfine size tuple fields Htuple Hfields.
  apply (tuple_fields_of_blocks _ statics size tuple fields);
    [ | exact Hfields ].
  intros tag block Hblock. apply forallb_Forall_among.
  destruct (block_of_cases _ _ _ _ Hblock) as [-> | (sym & -> & Hs)].
  - exact Htuple.
  - exact (statics_among_lookup _ _ statics sym _ Hfine Hs).
Qed.

(* The arguments that the closure of a call is called with. *)
Lemma called_arguments_bound (statics : static_env) (callee : value)
    (vs : list value) :
  value_statics_bound deleted statics callee ->
  Forall (value_statics_bound deleted statics) vs ->
  Forall (value_statics_bound deleted statics) (called_arguments callee vs).
Proof.
  intros Hcallee Hvs. unfold called_arguments.
  destruct callee; simpl; try exact Hvs.
  apply Forall_app. split; [ | exact Hvs ].
  exact (proj2 (proj1 (value_bound_partial statics _ _ _) Hcallee)).
Qed.

(* The code of a closure that mentions only what is bound is the same
   in the two static environments: code that the first has bound, the
   second has; and code that is declared deleted neither has. *)
Lemma codes_agree_hoist (statics : static_env) (closure : value)
    (called : code_id) (slots : list (value_slot * value)) :
  statics_known statics ->
  statics_fine statics ->
  value_statics_bound deleted statics closure ->
  closure_of statics closure = Some (called, slots) ->
  codes_agree eq statics (hoist_env statics) called.
Proof.
  intros [Scodes _] Hfine Hclosure Hcalled.
  destruct (closure_of_among _ _ statics closure called slots Hfine Hclosure
              Hcalled)
    as [Hbound _].
  unfold codes_agree. cbn [hoist_env static_code].
  destruct (lookup called (static_code statics)) as [c | ] eqn:Hlookup.
  - rewrite (Scodes called c Hlookup). auto.
  - rewrite among_app in Hbound.
    apply orb_true_iff in Hbound as [Hbound | Hdeleted].
    + unfold bound_code_ids in Hbound. rewrite <- is_bound_among in Hbound.
      unfold is_bound in Hbound. rewrite Hlookup in Hbound. discriminate.
    + rewrite (deleted_not_src called Hdeleted). exact I.
Qed.

Lemma shape_of_call_hoist (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  statics_known statics ->
  statics_fine statics ->
  value_statics_bound deleted statics callee ->
  Forall (value_statics_bound deleted statics) vs ->
  shapes_related eq
    (shape_of_call statics callee kind vs args_arity return_arity)
    (shape_of_call (hoist_env statics) callee kind vs args_arity
       return_arity).
Proof.
  intros S Hfine Hcallee Hvs.
  pose proof (closure_called_bound deleted statics callee Hcallee)
    as Hclosure.
  apply shape_of_call_related.
  - exact (closure_of_hoist statics _ S Hclosure).
  - intros called slots.
    exact (codes_agree_hoist statics _ called slots S Hfine Hclosure).
  - intros tuple rest p Hargs.
    apply eval_prim_static_values. intros sym [Hsym | [ ]].
    apply (lookup_bound_hoist statics sym S).
    pose proof (called_arguments_bound statics callee vs Hcallee Hvs)
      as Hall.
    rewrite Hargs, Hsym in Hall. exact (Forall_inv Hall).
Qed.

(* A call of a given shape, given what the two runs do with a partial
   application. *)
Lemma run_call_hoist (statics : static_env) (hp : heap) (rd : read_record)
    (return_partial return_partial' : value -> step_result)
    (shape shape' : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) :
  shapes_related eq shape shape' ->
  shape_values_all (value_statics_bound deleted statics) shape ->
  (∀ partial : value,
    value_statics_bound deleted statics partial ->
    result_hoisted (return_partial partial) (return_partial' partial)) ->
  result_hoisted
    (run_call return_partial shape return_arity return_to exn_to statics hp rd)
    (run_call return_partial' shape' return_arity (hoist_cont return_to)
       (hoist_cont exn_to) (hoist_env statics) hp rd).
Proof.
  intros Hshapes Hvalues Hpartial.
  destruct Hshapes
    as [ | closure c c' vs args_arity <-
       | closure c c' vs remaining args_arity remaining_arity <-
       | partial];
    simpl in Hvalues |- *.
  - reflexivity.
  - apply call_code_hoist.
  - exact (call_code_hoist closure c vs args_arity (code_result_arity c)
             (Over_application remaining remaining_arity return_arity
                return_to exn_to)
             exn_to statics hp rd).
  - exact (Hpartial partial Hvalues).
Qed.

Lemma call_function_hoist_from (statics : static_env) (hp : heap)
    (rd : read_record)
    (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) :
  statics_known statics ->
  statics_fine statics ->
  value_statics_bound deleted statics callee ->
  Forall (value_statics_bound deleted statics) vs ->
  (∀ partial : value,
    value_statics_bound deleted statics partial ->
    result_hoisted (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value (hoist_cont return_to) [partial]
         (hoist_env statics) hp rd)) ->
  result_hoisted
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity
       (hoist_cont return_to) (hoist_cont exn_to) (hoist_env statics) hp rd).
Proof.
  intros S Hfine Hcallee Hvs Hpartial. unfold call_function.
  apply run_call_hoist.
  - apply shape_of_call_hoist; assumption.
  - apply (shape_of_call_values_all _ (value_bound_partial statics));
      [ exact (tuple_fields_bound statics Hfine) | assumption | assumption ].
  - exact Hpartial.
Qed.

Lemma apply_cont_value_hoist (statics : static_env) (hp : heap)
    (rd : read_record)
    (c : cont_value) :
  statics_known statics ->
  statics_fine statics ->
  ∀ vs : list value,
    cont_known c ->
    cont_values_all (value_statics_bound deleted statics) c ->
    Forall (value_statics_bound deleted statics) vs ->
    result_hoisted (apply_cont_value c vs statics hp rd)
      (apply_cont_value (hoist_cont c) vs (hoist_env statics) hp rd).
Proof.
  intros S Hfine.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs Hknown Hvalues Hvs.
  - unfold result_hoisted. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | reflexivity ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | reflexivity ].
    simpl in Hknown. destruct Hknown as (_ & Hrec & _).
    unfold hoist_state. simpl_state. f_equal. f_equal.
    fold (hoist_conts def_conts).
    rewrite (rec_closures_hoist recursive def_vars def_conts Hrec).
    unfold hoist_conts. rewrite map_app. reflexivity.
  - reflexivity.
  - simpl in Hknown, Hvalues |- *. destruct Hvalues as [Hextra Hc].
    apply IH; [ exact Hknown | exact Hc | ].
    apply Forall_app. split; assumption.
  - simpl hoist_cont. rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ reflexivity | | reflexivity ].
    simpl in Hknown, Hvalues. destruct Hknown as [Kreturn _].
    destruct Hvalues as (Hremaining & Vreturn & _).
    apply call_function_hoist_from;
      [ exact S | exact Hfine | exact (Forall_inv Hvs) | exact Hremaining | ].
    intros partial Hpartial.
    apply IHreturn; [ exact Kreturn | exact Vreturn | ].
    apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

Lemma call_function_hoist (statics : static_env) (hp : heap) (rd : read_record)
    (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) :
  statics_known statics ->
  statics_fine statics ->
  value_statics_bound deleted statics callee ->
  Forall (value_statics_bound deleted statics) vs ->
  cont_known return_to ->
  cont_values_all (value_statics_bound deleted statics) return_to ->
  result_hoisted
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity
       (hoist_cont return_to) (hoist_cont exn_to) (hoist_env statics) hp rd).
Proof.
  intros S Hfine Hcallee Hvs Kreturn Vreturn.
  apply call_function_hoist_from; try assumption.
  intros partial Hpartial.
  apply apply_cont_value_hoist; try assumption.
  apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

(* A step of a state of the first run is a step of the second state, to
   the second state of the result. *)
Lemma hoist_step_components (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  hoist_inv (State e vars conts statics hp rd) ->
  result_hoisted (step (State e vars conts statics hp rd))
    (step
       (State (without_hoisted e) vars (hoist_conts conts)
          (hoist_env statics) hp rd)).
Proof.
  intros (W & Ws & (_ & Sc & _) & (Vvars & Vconts & Vstatics & _)
          & (K & Kc & S)).
  simpl_state.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [without_hoisted]; try rewrite !step_let_exec;
    autorewrite with step;
    cbn [expr_statics_in_scope expr_known] in Sc, K.
  - (* Let *)
    apply andb_true_iff in Sc as [Sd _].
    rewrite (exec_named_hoist statics hp rd vars d S Vvars Sd).
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
      [ | reflexivity ].
    destruct (kind_eqb (kind_of_value v) kind); reflexivity.
  - (* Let_static *)
    rewrite (eval_static_bindings_hoist vars bindings statics (proj1 K)
               (inv_no_rebinding code_ids_of_binding code_id_bound
                  bindings body vars conts statics hp rd_ W)
               (inv_no_rebinding symbols_of_binding symbol_bound
                  bindings body vars conts statics hp rd_ Ws)).
    destruct (eval_static_bindings vars bindings statics); reflexivity.
  - (* Let_cont_non_recursive *)
    reflexivity.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ reflexivity | ].
    unfold result_hoisted, hoist_state. simpl_state.
    rewrite (rec_closures_hoist handlers vars conts Hbinds).
    unfold hoist_conts. rewrite map_app. reflexivity.
  - (* Apply *)
    unfold apply_statics_in_scope in Sc.
    apply andb_true_iff in Sc as [Sc _].
    apply andb_true_iff in Sc as [Sc Sargs].
    apply andb_true_iff in Sc as [Scallee _].
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | reflexivity ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | reflexivity ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | reflexivity ].
    rewrite !resolve_hoist, <- with_extra_args_hoist.
    apply (call_function_hoist statics); [ exact S | exact Vstatics | | | | ].
    + exact (eval_simple_among _ _ vars _ callee Vvars Scallee Hcallee).
    + apply forallb_Forall_among.
      exact (eval_simples_among _ _ vars _ Vvars vs Sargs Hargs).
    + apply resolve_known, Kc.
    + apply resolve_values_all, Vconts.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | reflexivity ].
    unfold apply_continuation. rewrite resolve_hoist.
    apply (apply_cont_value_hoist statics _ rd_ _ S Vstatics).
    + apply resolve_known, Kc.
    + apply resolve_values_all, Vconts.
    + apply forallb_Forall_among.
      exact (eval_simples_among _ _ vars _ Vvars vs Sc Hargs).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | reflexivity ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | reflexivity ].
    destruct (find_arm i arms) as [ac | ]; reflexivity.
  - reflexivity.
Qed.

Definition hoisted_related (s s' : state) : Prop :=
  s' = hoist_state s ∧ hoist_inv s.

Lemma hoisted_related_step (s s' : state) :
  hoisted_related s s' ->
  result_related hoisted_related (step s) (step s').
Proof.
  intros [-> Hinv].
  pose proof
    (hoist_step_components (state_expr s) (state_vars s) (state_conts s)
       (state_statics s) (state_heap s) (state_reads s))
    as Hstep.
  rewrite <- (state_eta s) in Hstep. specialize (Hstep Hinv).
  pose proof (hoist_inv_step s) as Hkept.
  unfold hoist_state at 1. unfold result_hoisted in Hstep.
  destruct (step s) as [t | k vs | ]; rewrite Hstep; simpl.
  - split; [ reflexivity | exact (Hkept t Hinv eq_refl) ].
  - auto.
  - exact I.
Qed.

End Hoisted.

(* Where every symbol that the first static environment binds is bound
   first, to the same, none of its bindings is kept. *)
Lemma keep_values_none
    (src_values values : list (symbol * statically_allocated_value)) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s values = Some allocated ->
    lookup s src_values = Some allocated) ->
  keep_values src_values values = [].
Proof.
  intros Hbound. unfold keep_values.
  assert (Hall :
    ∀ binding : symbol * statically_allocated_value,
      In binding values -> kept_symbol src_values (fst binding) = false).
  { intros [s a] Hin. cbn [fst]. unfold kept_symbol, hoisted_symbol.
    assert (His : is_bound s values = true).
    { rewrite is_bound_among. apply among_In. apply in_map_iff.
      exists (s, a). split; [ reflexivity | exact Hin ]. }
    apply is_bound_lookup in His as [a' Ha'].
    apply negb_false_iff. apply is_bound_lookup. exists a'.
    exact (Hbound s a' Ha'). }
  clear Hbound. induction values as [ | binding values IH]; [ reflexivity | ].
  cbn [filter]. rewrite (Hall binding (or_introl eq_refl)). apply IH.
  intros b Hb. apply Hall. right. exact Hb.
Qed.

(* The simulation of a unit that passes, run from a static environment
   that holds no code, binds no name that the unit binds, and binds its
   symbols to things that mention only symbols that it binds; and from
   a heap whose cells hold values that mention only those symbols. The
   bindings taken to the head evaluate, with no variable,
   to a static environment; the first state of the unit is related to
   the unit without those bindings in that environment, which is what
   the unit with them first steps to. The symbols of the environment
   that the run starts from are, for the simulation, symbols bound
   first: the second run has them bound from its first state too. *)
Lemma hoist_statics_related_from (u : unit_interface) (e : expr)
    (statics : static_env) (hp : heap) (rd : read_record) :
  well_formed_unit u e = true ->
  statics_in_scope [] e = true ->
  interface_conts_not_rebound u e = true ->
  static_names_apart [] e = true ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  Forall
    (fun binding =>
       allocated_statics_among (map fst (static_values statics)) []
         (snd binding)
       = true)
    (static_values statics) ->
  heap_values_all
    (fun v =>
       value_statics_among (map fst (static_values statics)) [] v = true)
    hp ->
  ∃ (src_code : list (code_id * code))
    (src_values : list (symbol * statically_allocated_value)),
    (∀ id : code_id,
      among id (deleted_code_ids e) = true -> lookup id src_code = None)
    ∧ eval_static_bindings [] (hoisted_bindings e) statics
      = Some (Static_env src_code src_values)
    ∧ hoisted_related (deleted_code_ids e) src_code src_values
        (State e [] [] statics hp rd)
        (State (without_hoisted e) [] [] (Static_env src_code src_values)
           hp rd).
Proof.
  intros Hwf Hscope Hconts Hapart Hno_code Hfree Hamong Hheap.
  pose proof (well_formed_unit_statics_outside_loops_and_code u e Hwf)
    as Htoplevel.
  pose proof (well_formed_unit_code_ids_distinct u e Hwf) as Hcodes.
  pose proof (well_formed_unit_symbols_distinct u e Hwf) as Hsymbols.
  (* The names of the bindings of the unit, and of those taken out, are
     distinct. *)
  pose proof
    (toplevel_names_nodup code_ids_of_binding e
       (proj1 (distinct_NoDup _) Hcodes))
    as Htop_codes.
  pose proof
    (toplevel_names_nodup symbols_of_binding e
       (proj1 (distinct_NoDup _) Hsymbols))
    as Htop_symbols.
  assert (Hhoisted_codes :
    distinct (flat_map code_ids_of_binding (hoisted_bindings e)) = true).
  { apply distinct_NoDup. rewrite hoisted_bindings_filter.
    exact (nodup_flat_map_filter _ _ _ Htop_codes). }
  assert (Hhoisted_symbols :
    distinct (flat_map symbols_of_binding (hoisted_bindings e)) = true).
  { apply distinct_NoDup. rewrite hoisted_bindings_filter.
    exact (nodup_flat_map_filter _ _ _ Htop_symbols). }
  (* No name that the unit binds is bound in the first environment. *)
  assert (Hfresh_code :
    ∀ id : code_id, lookup id (static_code statics) = None).
  { intros id. rewrite Hno_code. reflexivity. }
  assert (Hunbound :
    ∀ s : symbol,
      In s (static_names symbols_of_binding e) ->
      lookup s (static_values statics) = None).
  { intros s Hs.
    pose proof (free_in_symbols statics e Hfree s Hs) as Hfalse.
    unfold symbol_bound in Hfalse.
    destruct (lookup s (static_values statics));
      [ discriminate Hfalse | reflexivity ]. }
  assert (Hfresh_symbols :
    ∀ s : symbol,
      among s (flat_map symbols_of_binding (hoisted_bindings e)) = true ->
      lookup s (static_values statics) = None).
  { intros s Hs. apply among_In in Hs. rewrite hoisted_bindings_filter in Hs.
    apply in_flat_map_filter, toplevel_names_in in Hs. exact (Hunbound s Hs). }
  destruct (bind_hoisted_from (hoisted_bindings e) statics
              (hoisted_bindings_are_hoisted e) Hhoisted_codes
              Hhoisted_symbols (fun id _ => Hfresh_code id) Hfresh_symbols)
    as (src_code & src_values & Hsrc & Hcode_in & _ & Hcode_only & Hsym_in
        & Hsym_old & Hsym_only).
  (* A code id that is declared deleted is not bound first. *)
  assert (Hdeleted :
    ∀ id : code_id,
      among id (deleted_code_ids e) = true -> lookup id src_code = None).
  { intros id Hid.
    destruct (lookup id src_code) as [c | ] eqn:Hlookup; [ | reflexivity ].
    exfalso.
    destruct (Hcode_only id) as [Hin | Hold];
      [ rewrite Hlookup; discriminate | | exact (Hold (Hfresh_code id)) ].
    apply among_In in Hin. rewrite hoisted_bindings_filter in Hin.
    apply in_flat_map_filter, toplevel_names_in, among_In in Hin.
    rewrite (deleted_not_bound [] e id Hapart Hid) in Hin. discriminate. }
  (* Every binding of the unit is known. *)
  assert (Hknown :
    Forall (binding_known src_code src_values) (toplevel_bindings e)).
  { apply Forall_forall. intros [id c | id | s const] Hin;
      cbn [binding_known].
    - apply Hcode_in. rewrite hoisted_bindings_filter. apply filter_In.
      split; [ exact Hin | reflexivity ].
    - exact I.
    - destruct (eval_static_const [] const) as [closed | ] eqn:Hclosed.
      + destruct (Hsym_in s const) as (closed' & Hclosed' & Hlookup).
        { rewrite hoisted_bindings_filter. apply filter_In.
          split; [ exact Hin | ]. cbn [is_hoisted]. rewrite Hclosed.
          reflexivity. }
        rewrite Hclosed in Hclosed'. injection Hclosed' as <-. exact Hlookup.
      + destruct (lookup s src_values) as [allocated | ] eqn:Hlookup;
          [ | reflexivity ].
        exfalso.
        destruct (Hsym_only s) as [Hin' | Hold];
          [ rewrite Hlookup; discriminate | | ].
        * apply among_In in Hin'. rewrite hoisted_bindings_filter in Hin'.
          refine (kept_symbol_not_hoisted s const (toplevel_bindings e) _
                    Htop_symbols Hin Hin').
          cbn [is_hoisted]. rewrite Hclosed. reflexivity.
        * apply Hold. apply Hunbound.
          apply (toplevel_names_in symbols_of_binding e).
          apply in_flat_map. exists (Bind_symbol s const).
          split; [ exact Hin | left; reflexivity ]. }
  assert (Hkeep : keep_values src_values (static_values statics) = []).
  { apply keep_values_none. exact Hsym_old. }
  exists src_code, src_values.
  refine (conj Hdeleted (conj Hsrc _)).
  split.
  { unfold hoist_state, hoist_env. simpl_state. rewrite Hkeep.
    reflexivity. }
  assert (Hbound :
    imports_bound (map fst (static_values statics)) statics).
  { split; [ exact Hno_code | ]. split; [ | exact Hamong ].
    intros s. exact (is_bound_among s (static_values statics)). }
  assert (Hscope' :
    statics_in_scope (map fst (static_values statics)) e = true).
  { exact (expr_scope_sub e [] _ [] [] (sub_nil _) (sub_nil _) Hscope). }
  destruct
    (initial_state_inv (map fst (static_values statics)) u e statics hp rd Hwf
       Hscope' Hbound Hheap)
    as [Hterms Hvalues].
  refine
    (conj
       (inv_from code_ids_of_binding code_id_bound e statics hp rd
          (free_in_code_ids statics e Hfree) Hno_code Htoplevel Hcodes)
       (conj
          (inv_from symbols_of_binding symbol_bound e statics
             hp rd (free_in_symbols statics e Hfree) Hno_code Htoplevel
             Hsymbols)
          (conj Hterms (conj Hvalues _)))).
  unfold known_inv. simpl_state.
  refine (conj (expr_known_of_bindings src_code src_values e Hknown)
            (conj I (conj _ _))).
  + intros id c H. rewrite Hfresh_code in H. discriminate H.
  + intros sym allocated H _. exact (Hsym_old sym allocated H).
Qed.

(* A unit that passes and the unit with those bindings first behave
   alike, each run from such a static environment and such a heap. *)
Lemma hoist_statics_equivalent_from (u : unit_interface) (e e' : expr)
    (statics : static_env) (hp : heap) (rd : read_record) :
  hoist_statics u e = Some e' ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  Forall
    (fun binding =>
       allocated_statics_among (map fst (static_values statics)) []
         (snd binding)
       = true)
    (static_values statics) ->
  heap_values_all
    (fun v =>
       value_statics_among (map fst (static_values statics)) [] v = true)
    hp ->
  equivalent (State e [] [] statics hp rd) (State e' [] [] statics hp rd).
Proof.
  unfold hoist_statics.
  destruct (well_formed_unit u e) eqn:Hwf; simpl; [ | discriminate ].
  destruct (statics_in_scope [] e) eqn:Hscope; simpl; [ | discriminate ].
  destruct (interface_conts_not_rebound u e) eqn:Hconts; simpl;
    [ | discriminate ].
  destruct (static_names_apart [] e) eqn:Hapart; [ | discriminate ].
  intros [= <-] Hno_code Hfree Hamong Hheap.
  destruct
    (hoist_statics_related_from u e statics hp rd Hwf Hscope Hconts Hapart
       Hno_code Hfree Hamong Hheap)
    as (src_code & src_values & Hdeleted & Hsrc & Hrel).
  apply (equivalent_trans _
           (State (without_hoisted e) [] [] (Static_env src_code src_values)
              hp rd)).
  - exact (bisimulation_equivalent
             (hoisted_related (deleted_code_ids e) src_code
                src_values)
             (hoisted_related_step (deleted_code_ids e) src_code
                src_values Hdeleted)
             _ _ Hrel).
  - apply equivalent_sym, step_equivalent.
    rewrite step_let_static, Hsrc. reflexivity.
Qed.

(* The static environment of the imports of a unit binds its symbols
   to things that mention nothing. *)
Lemma imports_statics_among (syms : list symbol)
    (imports : list (symbol * import_class)) :
  Forall
    (fun binding => allocated_statics_among syms [] (snd binding) = true)
    (static_values (imports_statics imports)).
Proof.
  unfold imports_statics. cbn [static_values].
  induction imports as [ | [s c] imports IH]; cbn [map]; constructor;
    [ reflexivity | exact IH ].
Qed.

Theorem hoist_statics_program_equivalent :
  hoist_statics_program_equivalent_statement.
Proof.
  intros u e e' Hhoist.
  exact
    (hoist_statics_equivalent_from u e e' empty_static_env empty_heap no_reads
           Hhoist eq_refl (free_in_empty e) (Forall_nil _)
           (heap_values_all_empty _)).
Qed.

Theorem hoist_statics_equivalent_imports :
  hoist_statics_equivalent_imports_statement.
Proof.
  intros imports u e e' Hhoist Hfree.
  exact (hoist_statics_equivalent_from u e e' (imports_statics imports)
           empty_heap no_reads Hhoist eq_refl Hfree
             (imports_statics_among _ imports)
           (heap_values_all_empty _)).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The two laws of the pass are proved from the definitions by
   [equivariance] (FlambdaEquivarianceTactic.v). What is hoisted is told
   by a test that evaluates a constant in the empty environment: its law
   is a law of the machine, by hand, and it is given to the tactic in the
   two forms in which computation leaves it, under the binder of a
   [filter]. *)

(* A constant has a value in the empty environment when the constant
   renamed has one: a renaming moves no symbol. *)
Lemma is_hoisted_renamed (r : renaming) (binding : static_binding) :
  is_hoisted (rename_static_binding r binding) = is_hoisted binding.
Proof.
  destruct binding as [id c | id | s const]; try reflexivity.
  cbn [rename_static_binding is_hoisted].
  rewrite <-
    (FlambdaEquivarianceProofs.eval_static_const_equivariant r [] const).
  reflexivity.
Qed.

Lemma filter_hoisted_renamed (r : renaming)
    (bindings : list static_binding) :
  filter (fun binding => is_hoisted (rename_static_binding r binding))
    bindings
  = filter is_hoisted bindings.
Proof. apply filter_ext. intros binding. apply is_hoisted_renamed. Qed.

Lemma filter_not_hoisted_renamed (r : renaming)
    (bindings : list static_binding) :
  filter
    (fun binding => negb (is_hoisted (rename_static_binding r binding)))
    bindings
  = filter (fun binding => negb (is_hoisted binding)) bindings.
Proof.
  apply filter_ext. intros binding. rewrite is_hoisted_renamed. reflexivity.
Qed.
#[local] Hint Rewrite filter_hoisted_renamed filter_not_hoisted_renamed
  : equivariance.

(* FlambdaValidatorEquivarianceProofs.v rewrites with the two laws by
   these names. *)
Lemma hoisted_bindings_renamed (r : renaming) (e : expr) :
  hoisted_bindings (rename_expr r e)
  = map (rename_static_binding r) (hoisted_bindings e).
Proof. equivariance. Qed.

Theorem hoisted_bindings_equivariant :
  hoisted_bindings_equivariant_statement.
Proof. exact hoisted_bindings_renamed. Qed.

Lemma without_hoisted_renamed (r : renaming) (e : expr) :
  rename_expr r (without_hoisted e) = without_hoisted (rename_expr r e).
Proof. equivariance. Qed.

Theorem without_hoisted_equivariant :
  without_hoisted_equivariant_statement.
Proof. exact without_hoisted_renamed. Qed.
