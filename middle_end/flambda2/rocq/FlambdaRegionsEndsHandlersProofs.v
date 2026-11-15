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

(* Lemmas for the proofs of the statements of FlambdaRegionsEndsSpec.v:
   the closures of handlers. The continuations of a state in which
   closures were made are as their entries say; the regions given at
   the parameters of a handler; and the state in which a handler is
   entered has the property, given what is asked of the frames there
   ([handler_entered]). Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsCallsSpec
  FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs FlambdaRegionsCheckedMarksProofs
  FlambdaRegionsCheckedOutlivesProofs FlambdaRegionsCheckedParamsProofs.
From OxCamlFlambda2 Require Import FlambdaRegionsEndsSpec
  FlambdaRegionsEndsBasicsProofs FlambdaRegionsEndsStableProofs.
Import ListNotations.

(* ======================================================================== *)
(* Bindings made of a list of handlers                                      *)
(* ======================================================================== *)

Lemma lookup_handlers_app {A B : Type} (g : continuation -> A -> B)
    (k : continuation) (l : list (continuation * A))
    (rest : list (continuation * B)) :
  lookup k (map (fun '(k', a) => (k', g k' a)) l ++ rest)
  = match lookup k l with
    | Some a => Some (g k a)
    | None => lookup k rest
    end.
Proof.
  induction l as [ | [k' a] l IH]; [ reflexivity | ].
  simpl. destruct (Pos.eqb k k') eqn:Hk; [ | exact IH ].
  apply Pos.eqb_eq in Hk. subst k'. reflexivity.
Qed.

Lemma lookup_rec_closures (k : continuation)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  lookup k (rec_closures handlers vars conts ++ conts)
  = match lookup k handlers with
    | Some h => Some (Closure h vars conts handlers)
    | None => lookup k conts
    end.
Proof.
  unfold rec_closures.
  exact
    (lookup_handlers_app (fun _ h => Closure h vars conts handlers) k handlers
       conts).
Qed.

(* ======================================================================== *)
(* The regions given at the parameters of a handler                         *)
(* ======================================================================== *)

Lemma eval_simples_nth (vars : env) (args : list simple) :
  ∀ (vs : list value) (i : nat) (s : simple),
    eval_simples vars args = Some vs ->
    nth_error args i = Some s ->
    ∃ v : value, nth_error vs i = Some v ∧ eval_simple vars s = Some v.
Proof.
  induction args as [ | a args IH]; intros vs i s Heval Hi.
  - destruct i; discriminate.
  - cbn in Heval.
    destruct (eval_simple vars a) as [v | ] eqn:Ha; cbn in Heval;
      [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ] eqn:Hargs; cbn in Heval;
      [ | discriminate ].
    injection Heval as <-.
    destruct i as [ | i]; cbn in *.
    + injection Hi as <-. exists v. split; [ reflexivity | exact Ha ].
    + exact (IH vs' i s eq_refl Hi).
Qed.

(* What is given at the parameters that are regions: each is given the
   variable of an open region, which is the argument at its place. *)
Lemma ends_region_args_given (open : list variable) (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : option (list simple))
    (given : list (variable * variable)),
    ends_region_args open params kinds args = Some given ->
    map fst given = region_params_of params kinds
    ∧ ∀ p r : variable,
        In (p, r) given ->
        among r open = true
        ∧ ∃ (l : list simple) (i : nat),
            args = Some l
            ∧ nth_error params i = Some p
            ∧ nth_error l i = Some (Var r).
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] args given H;
    cbn in H; try discriminate.
  - destruct args as [[ | a l] | ]; try discriminate; injection H as <-;
      (split; [ reflexivity | intros p r [] ]).
  - destruct args as [[ | a l] | ]; cbn in H; try discriminate.
    + destruct (ends_region_args open params kinds (Some l)) as [rest | ]
        eqn:Hrest; cbn in H; [ | discriminate ].
      destruct (IH kinds (Some l) rest Hrest) as [Hfst Hin].
      cbn [region_params_of]. destruct (kind_eqb kind Region).
      * destruct a as [r | sym | c]; try discriminate.
        destruct (among r open) eqn:Hr; [ | discriminate ].
        injection H as <-.
        split; [ cbn; f_equal; exact Hfst | ].
        intros p0 r0 [Heq | Hin0].
        -- injection Heq as <- <-. split; [ exact Hr | ].
           exists (Var r :: l), 0. repeat split; reflexivity.
        -- destruct (Hin p0 r0 Hin0) as (Hr0 & l' & i & Hl & Hp & Hi).
           injection Hl as <-. split; [ exact Hr0 | ].
           exists (Var r :: l), (S i). repeat split; assumption.
      * injection H as <-. split; [ exact Hfst | ].
        intros p0 r0 Hin0.
        destruct (Hin p0 r0 Hin0) as (Hr0 & l' & i & Hl & Hp & Hi).
        injection Hl as <-. split; [ exact Hr0 | ].
        exists (a :: l), (S i). repeat split; assumption.
    + destruct (ends_region_args open params kinds None) as [rest | ]
        eqn:Hrest; cbn in H; [ | discriminate ].
      destruct (IH kinds None rest Hrest) as [Hfst Hin].
      cbn [region_params_of]. destruct (kind_eqb kind Region);
        [ discriminate | ].
      injection H as <-. split; [ exact Hfst | ].
      intros p0 r0 Hin0.
      destruct (Hin p0 r0 Hin0) as (_ & l' & i & Hl & _). discriminate.
Qed.

(* A parameter that is a region is bound to what the variable given at
   its place is bound to. *)
(* The values that a handler is applied to are, where an argument is a
   variable, what the variable holds. It is asked of the variables
   only: an exception handler is applied to the exception, which the
   check has no argument for, and then to the extra arguments. *)
Definition args_given (vars : env) (args : option (list simple))
    (vs : list value) : Prop :=
  ∀ (l : list simple) (i : nat) (r : variable),
    args = Some l ->
    nth_error l i = Some (Var r) ->
    ∃ v : value, nth_error vs i = Some v ∧ lookup r vars = Some v.

Lemma args_given_eval (vars : env) (args : option (list simple))
    (vs : list value) :
  (∀ l : list simple, args = Some l -> eval_simples vars l = Some vs) ->
  args_given vars args vs.
Proof.
  intros Heval l i r Hl Hi.
  exact (eval_simples_nth vars l vs i (Var r) (Heval l Hl) Hi).
Qed.

Lemma args_given_none (vars : env) (vs : list value) :
  args_given vars None vs.
Proof. intros l i r Hl. discriminate. Qed.

Lemma args_given_raise (vars : env) (simples : list simple)
    (extra : list value) (v : value) :
  eval_simples vars simples = Some extra ->
  args_given vars (Some (unit_simple :: simples)) (v :: extra).
Proof.
  intros Heval l i r [= <-] Hi. destruct i as [ | i]; [ discriminate | ].
  exact (eval_simples_nth vars simples extra i (Var r) Heval Hi).
Qed.

Lemma given_bound (open : list variable) (params : list variable)
    (kinds : list flambda_kind) (args : option (list simple))
    (given : list (variable * variable)) (vars : env) (vs : list value)
    (def_vars handler_vars : env) :
  ends_region_args open params kinds args = Some given ->
  distinct params = true ->
  args_given vars args vs ->
  bind_params params vs def_vars = Some handler_vars ->
  ∀ p r : variable,
    In (p, r) given -> lookup p handler_vars = lookup r vars.
Proof.
  intros Hargs Hdistinct Heval Hbind p r Hin.
  destruct (ends_region_args_given open params kinds args given Hargs)
    as [_ Hgiven].
  destruct (Hgiven p r Hin) as (_ & l & i & Hl & Hp & Hi).
  destruct (Heval l i r Hl Hi) as (v & Hv & Hev).
  rewrite Hev.
  exact
    (proj2 (bind_params_nth params vs def_vars handler_vars Hdistinct Hbind) i
       p v Hp Hv).
Qed.

(* The marks of a list of open regions, by pairs. *)
Lemma combine_marked (vars : env) (l : list variable) :
  ∀ (ms : list mark) (r : variable) (m : mark),
    open_marks vars l = Some ms ->
    In (r, m) (combine l ms) ->
    In r l ∧ lookup r vars = Some (Value_symbol m).
Proof.
  intros ms r m H. apply open_marks_marked in H.
  induction H as [ | x m0 l ms Hx _ IH]; intros Hin; [ contradiction | ].
  cbn in Hin. destruct Hin as [Heq | Hin].
  - injection Heq as <- <-. split; [ left; reflexivity | exact Hx ].
  - destruct (IH Hin) as [H1 H2]. split; [ right; exact H1 | exact H2 ].
Qed.

Lemma marked_combine (vars : env) (l : list variable) :
  ∀ (ms : list mark) (r : variable) (m : mark),
    open_marks vars l = Some ms ->
    In r l ->
    lookup r vars = Some (Value_symbol m) ->
    In (r, m) (combine l ms).
Proof.
  intros ms r m H. apply open_marks_marked in H.
  induction H as [ | x m0 l ms Hx _ IH]; intros Hin Hr; [ contradiction | ].
  cbn. destruct Hin as [-> | Hin].
  - left. unfold marked in Hx. rewrite Hr in Hx. injection Hx as <-.
    reflexivity.
  - right. exact (IH Hin Hr).
Qed.

Lemma combine_map_l {A B C : Type} (f : A -> C) (l : list A) :
  ∀ (l' : list B) (a : A) (b : B),
    In (a, b) (combine l l') -> In (f a, b) (combine (map f l) l').
Proof.
  induction l as [ | x l IH]; intros [ | y l'] a b Hin; cbn in *;
    try contradiction.
  destruct Hin as [[= <- <-] | Hin];
    [ left; reflexivity | right; exact (IH l' a b Hin) ].
Qed.

(* The marks of the open regions of a handler that is entered are those
   of the regions that the jump names for them. *)
Lemma entered_open_marks (handler_vars vars : env)
    (given : list (variable * variable)) (opens : list variable) :
  (∀ r : variable,
     In r opens ->
     lookup r handler_vars = lookup (region_at_jump given r) vars) ->
  open_marks handler_vars opens
  = open_marks vars (map (region_at_jump given) opens).
Proof.
  induction opens as [ | r opens IH]; intros H; [ reflexivity | ].
  cbn. rewrite (H r (or_introl eq_refl)).
  rewrite (IH (fun r' Hin => H r' (or_intror Hin))). reflexivity.
Qed.

Lemma region_params_of_in (params : list variable) :
  ∀ (kinds : list flambda_kind) (r : variable),
    In r (region_params_of params kinds) -> In r params.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] r Hin;
    cbn in Hin; try contradiction.
  destruct (kind_eqb kind Region).
  - destruct Hin as [<- | Hin]; [ left; reflexivity | ].
    right. exact (IH kinds r Hin).
  - right. exact (IH kinds r Hin).
Qed.

(* A variable of a region that was in scope is not a parameter of a
   handler whose parameters are new. *)
Lemma new_not_param (regions params : list variable) (r : variable) :
  forallb (fun p => negb (among p regions)) params = true ->
  among r regions = true ->
  among r params = false.
Proof.
  intros Hnew Hregion. destruct (among r params) eqn:Hp; [ | reflexivity ].
  pose proof (proj1 (forallb_forall _ _) Hnew r (proj1 (among_In _ _) Hp))
    as Hneg.
  cbn beta in Hneg. rewrite Hregion in Hneg. discriminate.
Qed.

Lemma in_combine_exists {A B : Type} (l : list A) :
  ∀ (l' : list B) (b : B),
    length l' = length l -> In b l' -> ∃ a : A, In (a, b) (combine l l').
Proof.
  induction l as [ | a l IH]; intros [ | b' l'] b Hlength Hin;
    cbn in *; try contradiction; try discriminate.
  destruct Hin as [<- | Hin].
  - exists a. left. reflexivity.
  - injection Hlength as Hlength.
    destruct (IH l' b Hlength Hin) as [a' Ha']. exists a'. right. exact Ha'.
Qed.

(* ======================================================================== *)
(* A closure, taken apart                                                   *)
(* ======================================================================== *)

Section Handlers.

Variable cert : certificate.
Variable opts : ends_options.
Variable ret : continuation.
Variable bottom : symbol.

Lemma cont_ends_handler_inv (rs : region_state) (u : ends_user)
    (params : list variable) (kinds : list flambda_kind)
    (entry : handler_entry) (is_exn : bool) (c : cont_value) :
  cont_ends cert opts ret bottom rs u (Ends_handler params kinds entry is_exn)
    c ->
  ∃ (metadata : handler_metadata) (handler : expr) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (es : ends_state) (base : list (continuation * ends_cont)),
    c = Closure (Handler params metadata handler) vars conts recursive
    ∧ kinds = hm_params_kinds metadata
    ∧ is_exn = enters_stale opts (Handler params metadata handler)
    ∧ ends_made_at cert opts u vars recursive es base
    ∧ ends_expr cert opts (ends_handler_state es params kinds entry is_exn)
        handler
      = true
    ∧ distinct params = true
    ∧ forallb (fun p => negb (among p (es_regions es))) params = true
    ∧ distinct (he_open entry) = true
    ∧ forallb
        (fun r =>
           among r (region_params_of params kinds) || among r (es_open es))
        (he_open entry)
      = true
    ∧ (∀ (k : continuation) (ec : ends_cont),
         lookup k base = Some ec ->
         cont_ends cert opts ret bottom rs
           (Ends_user vars (es_regions es) (es_tries es) (eu_floor u)) ec
           (resolve k conts))
    ∧ (∀ k : continuation, lookup k base = None -> lookup k conts = None).
Proof.
  intros Hok. inversion Hok; subst.
  eexists _, _, _, _, _, _, _.
  repeat (split; [ first [ reflexivity | eassumption ] | ]).
  assumption.
Qed.

(* ======================================================================== *)
(* The continuations of a state in which closures were made                 *)
(* ======================================================================== *)

(* The handlers of a recursive group, each closed over the group, and
   the continuations that were in scope: as the entries say, for a
   user that extends the one that the closures were made for. *)
Lemma closures_conts_ends (rs : region_state) (u u' : ends_user) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (es : ends_state) (base : list (continuation * ends_cont)) :
  ends_made_at cert opts u vars recursive es base ->
  (∀ (k : continuation) (ec : ends_cont),
     lookup k base = Some ec ->
     cont_ends cert opts ret bottom rs
       (Ends_user vars (es_regions es) (es_tries es) (eu_floor u)) ec
       (resolve k conts)) ->
  (∀ k : continuation, lookup k base = None -> lookup k conts = None) ->
  ends_user_extends
    (Ends_user vars (es_regions es) (es_tries es) (eu_floor u)) u' ->
  conts_ends cert opts ret bottom rs u' (es_conts es)
    (rec_closures recursive vars conts ++ conts).
Proof.
  intros Hmade Hbase Hnone Hextends.
  pose proof Hmade as (Hconts & Hhandlers & Hopen & Htries & Hregions & Htry).
  destruct Hextends as (Hext_regions & Hext_tries & Hext_floor).
  cbn [eu_vars eu_regions eu_tries eu_floor] in *.
  assert (Hmade' : ends_made_at cert opts u' vars recursive es base).
  { split; [ exact Hconts | ]. split; [ exact Hhandlers | ].
    split; [ exact Hopen | ]. split; [ exact Htries | ].
    split; [ exact Hext_regions | exact Hext_tries ]. }
  assert (Hbase' :
            ∀ (k : continuation) (ec : ends_cont),
              lookup k base = Some ec ->
              cont_ends cert opts ret bottom rs
                (Ends_user vars (es_regions es) (es_tries es) (eu_floor u'))
                ec (resolve k conts)).
  { rewrite Hext_floor. exact Hbase. }
  rewrite Hconts. split.
  - intros k ec Hk. rewrite lookup_handlers_app in Hk.
    unfold resolve. rewrite lookup_rec_closures.
    destruct (lookup k recursive) as [h | ] eqn:Hrec.
    + injection Hk as <-.
      destruct h as [params metadata handler]. unfold ends_cont_of.
      destruct (lookup k (cert_handlers cert)) as [entry | ] eqn:Hentry;
        [ | apply Ends_ok_unreachable ].
      pose proof
        (proj1 (forallb_forall _ _) Hhandlers _ (lookup_in _ _ _ Hrec))
        as Hhandler.
      cbn in Hhandler. rewrite Hentry in Hhandler.
      rewrite !andb_true_iff in Hhandler.
      destruct Hhandler as [[[[H1 H2] H3] H4] H5].
      exact
        (Ends_ok_handler cert opts ret bottom rs u' params metadata handler
           entry vars conts recursive es base Hmade' H5 H1 H2 H3 H4 Hbase'
           Hnone).
    + apply (user_cont_ends cert opts ret bottom rs
               (Ends_user vars (es_regions es) (es_tries es) (eu_floor u)) u').
      * split; [ exact Hext_regions | ].
        split; [ exact Hext_tries | exact Hext_floor ].
      * change
          (match lookup k conts with
           | Some c => c
           | None => Outer k
           end)
          with (resolve k conts).
        exact (Hbase k ec Hk).
  - intros k Hk. rewrite lookup_handlers_app in Hk.
    rewrite lookup_rec_closures.
    destruct (lookup k recursive) as [h | ]; [ discriminate | ].
    exact (Hnone k Hk).
Qed.

(* ======================================================================== *)
(* A handler is entered                                                     *)
(* ======================================================================== *)

(* What a variable of an open region of the handler is bound to: what
   the region that the jump names for it is bound to, for the user. *)
Lemma entered_lookup (u : ends_user) (open : list variable)
    (params : list variable) (kinds : list flambda_kind)
    (args : option (list simple)) (given : list (variable * variable))
    (vs : list value) (def_vars handler_vars : env)
    (recursive : list (continuation * continuation_handler))
    (es : ends_state) (base : list (continuation * ends_cont))
    (entry : handler_entry) :
  ends_region_args open params kinds args = Some given ->
  distinct params = true ->
  args_given (eu_vars u) args vs ->
  bind_params params vs def_vars = Some handler_vars ->
  ends_made_at cert opts u def_vars recursive es base ->
  forallb (fun p => negb (among p (es_regions es))) params = true ->
  forallb
    (fun r => among r (region_params_of params kinds) || among r (es_open es))
    (he_open entry)
  = true ->
  ∀ r : variable,
    In r (he_open entry) ->
    lookup r handler_vars = lookup (region_at_jump given r) (eu_vars u).
Proof.
  intros Hargs Hdistinct Heval Hbind Hmade Hnew Hin_open r Hr.
  destruct Hmade as (_ & _ & Hopen & _ & Hregions & _).
  unfold region_at_jump. destruct (lookup r given) as [r' | ] eqn:Hg.
  - exact
      (given_bound open params kinds args given (eu_vars u) vs def_vars
         handler_vars Hargs Hdistinct Heval Hbind r r' (lookup_in _ _ _ Hg)).
  - pose proof (proj1 (forallb_forall _ _) Hin_open r Hr) as Hr'.
    apply orb_true_iff in Hr'. destruct Hr' as [Hparam | Hopen_r].
    + exfalso. apply (lookup_none_not_in _ _ Hg).
      change (In r (map fst given)).
      rewrite
        (proj1 (ends_region_args_given open params kinds args given Hargs)).
      apply among_In. exact Hparam.
    + assert (Hregion : among r (es_regions es) = true).
      { unfold FlambdaMapSpec.all_among in Hopen.
        exact
          (proj1 (forallb_forall _ _) Hopen r (proj1 (among_In _ _) Hopen_r)). }
      rewrite
        (bind_params_lookup_other r params vs def_vars handler_vars
           (new_not_param _ _ r Hnew Hregion) Hbind).
      symmetry. exact (proj1 (Hregions r Hregion)).
Qed.

(* The state in which a handler is entered has the property. The
   handler is a closure that is as an entry says, for the user that
   enters it, whose open regions are placed; the regions that it is
   entered with, in the names of the user, are open there and bound to
   marks that are placed; and the frames are tight for them. *)
Lemma handler_entered (rs' : region_state) (vars_j : env) (es_j : ends_state)
    (ms_j : list mark) (floor : symbol) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (def_vars : env)
    (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (entry : handler_entry) (args : option (list simple)) (vs : list value)
    (handler_vars : env) (named : list variable) (ms_h : list mark) :
  frames_ordered rs' = true ->
  code_ends cert opts rs' ->
  state_expr (rs_state rs') = handler ->
  state_vars (rs_state rs') = handler_vars ->
  state_conts (rs_state rs')
  = rec_closures recursive def_vars def_conts ++ def_conts ->
  Pos.leb bottom floor = true ->
  Pos.leb floor (rs_next rs') = true ->
  ends_placed vars_j (rs_frames rs') es_j ms_j floor ->
  cont_ends cert opts ret bottom rs'
    (Ends_user vars_j (es_regions es_j) (es_tries es_j) floor)
    (Ends_handler params (hm_params_kinds metadata) entry
       (enters_stale opts (Handler params metadata handler)))
    (Closure (Handler params metadata handler) def_vars def_conts recursive) ->
  entry_at_jump (es_open es_j) params (hm_params_kinds metadata) entry args
  = Some named ->
  args_given vars_j args vs ->
  bind_params params vs def_vars = Some handler_vars ->
  open_marks vars_j named = Some ms_h ->
  marks_placed (rs_frames rs') ms_h = true ->
  (∀ r : variable, In r named -> In r (es_open es_j)) ->
  frames_tight (rs_frames rs') floor
    (stale_stop (enters_stale opts (Handler params metadata handler)) floor
       ms_h)
    ms_h ->
  state_ends cert opts ret bottom rs'.
Proof.
  intros Hordered Hcode Hexpr Hvars Hconts Hbottom Hnext Hplaced Hcont Hnamed
    Heval Hbind Hmarks Hplaced_h Hsub Htight.
  destruct (cont_ends_handler_inv rs' _ params _ entry _ _ Hcont)
    as (metadata' & handler' & vars' & conts' & recursive' & es & base & Hc
        & _ & _ & Hmade & Hcheck & Hdistinct & Hnew & Hopen_distinct
        & Hopen_in & Hbase & Hnone).
  injection Hc as Hm Hh Hv Hcs Hr. subst metadata' handler' vars' conts'.
  subst recursive'.
  unfold entry_at_jump in Hnamed.
  destruct
    (ends_region_args (es_open es_j) params (hm_params_kinds metadata) args)
    as [given | ] eqn:Hgiven; cbn in Hnamed; [ | discriminate ].
  injection Hnamed as <-.
  pose proof Hplaced
    as (Hmarks_j & Hplaced_j & Hfloor_j & Hopen_j & Htries_j & Hkinds_j).
  pose proof Hmade as (_ & _ & Hopen_es & Htries_es & Hregions_es & Htry_es).
  cbn [eu_vars eu_regions eu_tries eu_floor] in *.
  unfold FlambdaMapSpec.all_among in Hopen_es, Htries_es.
  assert (Hlookup :
            ∀ r : variable,
              In r (he_open entry) ->
              lookup r handler_vars = lookup (region_at_jump given r) vars_j).
  { exact
      (entered_lookup
         (Ends_user vars_j (es_regions es_j) (es_tries es_j) floor)
         (es_open es_j) params (hm_params_kinds metadata) args given vs
         def_vars handler_vars recursive es base entry Hgiven Hdistinct Heval
         Hbind Hmade Hnew Hopen_in). }
  assert (Hmarks_h : open_marks handler_vars (he_open entry) = Some ms_h).
  { rewrite
      (entered_open_marks handler_vars vars_j given (he_open entry) Hlookup).
    exact Hmarks. }
  assert (Hpairs :
            ∀ (r : variable) (m : mark),
              In (r, m) (combine (he_open entry) ms_h) ->
              In (region_at_jump given r, m) (combine (es_open es_j) ms_j)).
  { intros r m Hin.
    pose proof
      (combine_map_l (region_at_jump given) (he_open entry) ms_h r m Hin)
      as Hin'.
    destruct (combine_marked vars_j _ ms_h _ m Hmarks Hin')
      as [Hin_named Hmark].
    exact
      (marked_combine vars_j (es_open es_j) ms_j _ m Hmarks_j
         (Hsub _ Hin_named) Hmark). }
  exists
    (ends_handler_state es params (hm_params_kinds metadata) entry
       (enters_stale opts (Handler params metadata handler))).
  split; [ exact Hordered | ]. split; [ exact Hcode | ].
  exists ms_h, floor.
  split; [ rewrite Hexpr; exact Hcheck | ].
  split; [ exact Hbottom | ]. split; [ exact Hnext | ]. split.
  - split; [ | exact Htight ].
    rewrite Hvars. cbn [ends_handler_state es_open es_tries es_regions].
    split; [ exact Hmarks_h | ]. split; [ exact Hplaced_h | ]. split.
    { apply forallb_forall. intros m Hm.
      destruct
        (in_combine_exists (he_open entry) ms_h m
           (open_marks_length _ _ _ Hmarks_h) Hm)
        as [r Hin].
      exact
        (proj1 (forallb_forall _ _) Hfloor_j m
           (in_combine_r _ _ _ _ (Hpairs r m Hin))). }
    split.
    { unfold FlambdaMapSpec.all_among. apply forallb_forall. intros r Hr.
      cbv beta iota delta [es_regions es_open es_tries ends_handler_state].
      rewrite among_app.
      pose proof (proj1 (forallb_forall _ _) Hopen_in r Hr) as H.
      cbn beta in H. apply orb_true_iff in H. apply orb_true_iff.
      destruct H as [H | H]; [ left; exact H | right ].
      exact (proj1 (forallb_forall _ _) Hopen_es r (proj1 (among_In _ _) H)). }
    split.
    { unfold FlambdaMapSpec.all_among. apply forallb_forall. intros r Hr.
      cbv beta iota delta [es_regions es_open es_tries ends_handler_state].
      rewrite among_app. apply orb_true_iff. right.
      exact (proj1 (forallb_forall _ _) Htries_es r Hr). }
    intros r m Hin Htry.
    assert (Hregion : among r (es_regions es) = true).
    { exact
        (proj1 (forallb_forall _ _) Htries_es r (proj1 (among_In _ _) Htry)). }
    assert (Hself : region_at_jump given r = r).
    { unfold region_at_jump.
      destruct (lookup r given) as [r' | ] eqn:Hg; [ | reflexivity ].
      exfalso.
      assert (Hparam :
                In r (region_params_of params (hm_params_kinds metadata))).
      { rewrite <- (proj1 (ends_region_args_given _ _ _ _ _ Hgiven)).
        apply in_map_iff. exists (r, r').
        split; [ reflexivity | exact (lookup_in _ _ _ Hg) ]. }
      pose proof
        (new_not_param _ _ r Hnew Hregion) as Hnot.
      rewrite (proj2 (among_In _ _) (region_params_of_in _ _ _ Hparam)) in Hnot.
      discriminate. }
    pose proof (Hpairs r m Hin) as Hin'. rewrite Hself in Hin'.
    exact (Hkinds_j r m Hin' (Htry_es r Htry)).
  - rewrite Hvars, Hconts.
    apply
      (closures_conts_ends rs'
         (Ends_user vars_j (es_regions es_j) (es_tries es_j) floor) _ def_vars
         def_conts recursive es base Hmade Hbase Hnone).
    unfold ends_user_extends.
    cbv beta iota delta
      [eu_floor eu_vars eu_regions eu_tries ends_handler_state es_regions
         es_tries].
    split; [ | split; [ | reflexivity ] ].
    + intros r Hr. split.
      * exact
          (bind_params_lookup_other r params vs def_vars handler_vars
             (new_not_param _ _ r Hnew Hr) Hbind).
      * rewrite among_app. apply orb_true_iff. right. exact Hr.
    + intros r Hr. exact Hr.
Qed.

End Handlers.
