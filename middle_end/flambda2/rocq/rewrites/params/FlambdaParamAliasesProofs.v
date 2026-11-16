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

(* Proofs of the statements of FlambdaParamAliasesSpec.v. Nothing here
   needs review.

   The pass changes a handler on the strength of what the applications
   in its scope pass, so it is not a rewrite of two expressions on their
   own. The term and its result are run side by side
   (FlambdaSimulationProofs.stuttering_refines): the second run takes the
   steps of the first and, where a handler starts, those of the [Let]s
   that the pass put there. The relation between the two states
   ([state_aliased]) is:

   - the second expression is the first with, at some [Let_cont]s, some
     parameters bound again at the start of the handler to what every
     application gives them ([aliased]). The relation keeps, for each
     continuation in scope, what its parameters are always given and,
     inside the handler of a loop, its parameters ([fates]); it asks that
     every application fits ([args_fit]), and that no binder in scope
     binds a name that this depends on ([unfrozen]);
   - the variables of the two states are bound alike;
   - each continuation name stands, in the two states, for the same
     thing or for closures whose handlers are so related
     ([cont_aliased]); and what a parameter is always given means, under
     the variables of the first state, what it means under those of the
     closure, which is also what the parameter itself is bound to inside
     the handler of a loop ([held]).

   Applying two such closures to the same values gives, once the second
   has run the [Let]s, states so related ([apply_cont_aliased]): each
   [Let] binds its parameter to the value it has already. That value is
   of the kind of the [Let], which is the kind that the handler gives
   the parameter ([kinds_fit]): the handler has just been applied to it,
   and has a step only on values of its kinds
   (FlambdaSemanticsSpec.apply_cont_value). The second handler gives the
   kinds of the first. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaSimulationProofs
  FlambdaSinkContsProofs FlambdaUnusedParamsSpec FlambdaUnusedParamsProofs
  FlambdaParamAliasesSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known of the continuations in scope                              *)
(* ======================================================================== *)

(* What each parameter of a continuation is always given ([None] where
   that is not one thing) and, inside the handler of a loop, the
   parameters of the loop: there an application may pass the parameter
   itself. Elsewhere the second list is empty. *)
Definition fate : Type := (list (option simple) * list variable)%type.

(* For the continuations in scope, the most recently bound first. Of one
   that is not listed nothing is known. *)
Definition fates : Type := list (continuation * fate).

Definition fate_of (K : fates) (k : continuation) : fate :=
  match lookup k K with
  | Some f => f
  | None => ([], [])
  end.

Definition alias_at (f : fate) (i : nat) (s : simple) : Prop :=
  nth_error (fst f) i = Some (Some s).

Definition no_alias (f : fate) : Prop :=
  ∀ (i : nat) (s : simple), ~ alias_at f i s.

(* The names whose meaning the fate depends on. *)
Definition frozen (f : fate) (x : variable) : Prop :=
  ∃ (i : nat) (s : simple),
    alias_at f i s ∧ (s = Var x ∨ nth_error (snd f) i = Some x).

Definition unfrozen (K : fates) (x : variable) : Prop :=
  ∀ k : continuation, ~ frozen (fate_of K k) x.

(* The arguments of an application are what the fate says. *)
Definition args_fit (f : fate) (args : list simple) : Prop :=
  ∀ (i : nat) (s : simple),
    alias_at f i s ->
    ∃ arg : simple,
      nth_error args i = Some arg
      ∧ (arg = s ∨ ∃ p : variable, nth_error (snd f) i = Some p ∧ arg = Var p).

(* What is asked of the parameters where some are bound again: they are
   distinct, and what one is bound to is none of them. *)
Definition aliases_fit (aliases : list (option simple))
    (params : list variable) : Prop :=
  ∀ (i : nat) (s : simple),
    nth_error aliases i = Some (Some s) ->
    NoDup params ∧ ∀ x : variable, s = Var x -> x ∉ params.

Lemma fate_of_head (K : fates) (k : continuation) (f : fate) :
  fate_of ((k, f) :: K) k = f.
Proof. unfold fate_of. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma fate_of_other (K : fates) (k k' : continuation) (f : fate) :
  k' <> k -> fate_of ((k, f) :: K) k' = fate_of K k'.
Proof.
  intros Hne. unfold fate_of. simpl.
  destruct (Pos.eqb_spec k' k); [ contradiction | reflexivity ].
Qed.

Lemma fate_of_pair (k1 k2 k : continuation) (f1 f2 : fate) :
  fate_of [(k1, f1); (k2, f2)] k
  = if Pos.eqb k k1 then f1 else if Pos.eqb k k2 then f2 else ([], []).
Proof.
  unfold fate_of. simpl. destruct (Pos.eqb k k1); [ reflexivity | ].
  destruct (Pos.eqb k k2); reflexivity.
Qed.

Lemma resolve_pair (k1 k2 k : continuation) (c1 c2 : cont_value) :
  resolve k [(k1, c1); (k2, c2)]
  = if Pos.eqb k k1 then c1 else if Pos.eqb k k2 then c2 else Outer k.
Proof.
  unfold resolve. simpl. destruct (Pos.eqb k k1); [ reflexivity | ].
  destruct (Pos.eqb k k2); reflexivity.
Qed.

(* The fates that the handlers of a recursive [Let_cont] add where
   nothing is found of them. *)
Definition plain_fates (handlers : list (continuation * continuation_handler))
    : fates :=
  map (fun '(k, _) => (k, (([], []) : fate))) handlers.

Lemma fate_of_plain (handlers : list (continuation * continuation_handler))
    (K : fates) (k : continuation) :
  fate_of (plain_fates handlers ++ K) k
  = match lookup k handlers with
    | Some _ => ([], [])
    | None => fate_of K k
    end.
Proof.
  unfold fate_of.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma no_alias_nil (alts : list variable) : no_alias ([], alts).
Proof.
  intros i s Halias. unfold alias_at in Halias. simpl in Halias.
  destruct i; discriminate Halias.
Qed.

Lemma no_alias_unfrozen (f : fate) (x : variable) :
  no_alias f -> ~ frozen f x.
Proof. intros Hno (i & s & Halias & _). exact (Hno i s Halias). Qed.

(* ======================================================================== *)
(* The [Let]s at the start of a handler                                     *)
(* ======================================================================== *)

Lemma bind_aliases_nil (params : list variable) (kinds : list flambda_kind)
    (handler : expr) :
  bind_aliases params kinds [] handler = handler.
Proof. destruct params, kinds; reflexivity. Qed.

Lemma bind_aliases_binds_statics (params : list variable) :
  ∀ (kinds : list flambda_kind) (aliases : list (option simple))
    (handler : expr),
    binds_statics (bind_aliases params kinds aliases handler)
    = binds_statics handler.
Proof.
  induction params as [ | p params IH]; intros kinds aliases handler;
    [ reflexivity | ].
  destruct kinds as [ | kind kinds]; [ reflexivity | ].
  destruct aliases as [ | [s | ] aliases]; [ reflexivity | | ]; simpl;
    apply IH.
Qed.

(* Where each parameter to bind again has the value of what it is bound
   to already, and that value is of the kind of its [Let], the [Let]s
   run and leave the variables as they were. *)
Lemma run_bind_aliases (params : list variable) :
  ∀ (kinds : list flambda_kind) (aliases : list (option simple))
    (handler : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record),
    (∀ (i : nat) (p : variable) (s : simple),
      nth_error params i = Some p ->
      nth_error aliases i = Some (Some s) ->
      ∃ v : value,
        eval_simple vars s = Some v ∧ lookup p vars = Some v
        ∧ ∀ kind : flambda_kind,
            nth_error kinds i = Some kind -> kind_of_value v = kind) ->
    ∃ (n : nat) (vars' : env),
      run n
        (State (bind_aliases params kinds aliases handler) vars conts statics
           hp rd)
      = Step (State handler vars' conts statics hp rd)
      ∧ ∀ x : variable, lookup x vars' = lookup x vars.
Proof.
  induction params as [ | p params IH];
    intros kinds aliases handler vars conts statics hp rd Hgiven.
  - exists 0, vars. split; reflexivity.
  - destruct kinds as [ | kind kinds];
      [ exists 0, vars; split; reflexivity | ].
    destruct aliases as [ | alias aliases];
      [ exists 0, vars; split; reflexivity | ].
    simpl. destruct alias as [s | ].
    + destruct (Hgiven 0 p s eq_refl eq_refl) as (v & Hs & Hp & Hkind).
      specialize (Hkind kind eq_refl).
      destruct
        (IH kinds aliases handler ((p, v) :: vars) conts statics hp rd)
        as (n & vars' & Hrun & Hsame).
      { intros i p' s' Hp' Hs'.
        destruct (Hgiven (S i) p' s' Hp' Hs') as (v' & Hs'v & Hp'v & Hk').
        exists v'. split; [ | split; [ | exact Hk' ] ].
        - rewrite (eval_simple_agree vars ((p, v) :: vars) s');
            [ exact Hs'v | ].
          intros x _. simpl.
          destruct (Pos.eqb_spec x p) as [-> | _]; [ exact Hp | reflexivity ].
        - simpl. destruct (Pos.eqb_spec p' p) as [-> | _];
            [ congruence | exact Hp'v ]. }
      exists (S n), vars'. split.
      * rewrite run_succ, step_let_simple. simpl.
        rewrite Hs, Hkind, kind_eqb_refl. exact Hrun.
      * intros x. rewrite Hsame. simpl.
        destruct (Pos.eqb_spec x p) as [-> | _];
          [ symmetry; exact Hp | reflexivity ].
    + apply IH. intros i p' s' Hp' Hs'. exact (Hgiven (S i) p' s' Hp' Hs').
Qed.

(* ======================================================================== *)
(* Values of arguments and of parameters                                    *)
(* ======================================================================== *)

Lemma eval_simples_nth (vars : env) (args : list simple) :
  ∀ (vs : list value) (i : nat) (arg : simple),
    eval_simples vars args = Some vs ->
    nth_error args i = Some arg ->
    ∃ v : value, nth_error vs i = Some v ∧ eval_simple vars arg = Some v.
Proof.
  induction args as [ | s args IH]; intros vs i arg Hvs Harg;
    [ destruct i; discriminate Harg | ].
  simpl in Hvs.
  destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
  destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
    [ | discriminate ].
  injection Hvs as <-. destruct i as [ | i]; simpl in Harg |- *.
  - injection Harg as <-. exists v. split; [ reflexivity | exact Hs ].
  - exact (IH rest i arg eq_refl Harg).
Qed.

(* Binding the parameters under variables bound alike gives variables
   bound alike. *)
Lemma bind_params_alike (params : list variable) (vs : list value)
    (vars vars' bound : env) :
  (∀ x : variable, lookup x vars = lookup x vars') ->
  bind_params params vs vars = Some bound ->
  ∃ bound' : env,
    bind_params params vs vars' = Some bound'
    ∧ ∀ x : variable, lookup x bound = lookup x bound'.
Proof.
  intros Hsame Hbind.
  destruct (bind_params_defined params vs vars vars' bound Hbind)
    as [bound' Hbind'].
  exists bound'. split; [ exact Hbind' | ]. intros x. symmetry.
  apply (bind_params_agree params vs vars vars' bound bound' Hbind Hbind').
  right. symmetry. apply Hsame.
Qed.

(* A parameter is bound to the value at its position, where the
   parameters are distinct. *)
Lemma bind_params_nth (params : list variable) :
  ∀ (i : nat) (p : variable) (v : value) (vs : list value)
    (vars bound : env),
    NoDup params ->
    nth_error params i = Some p ->
    nth_error vs i = Some v ->
    bind_params params vs vars = Some bound ->
    lookup p bound = Some v.
Proof.
  induction params as [ | p0 params IH];
    intros i p v vs vars bound Hnodup Hp Hv Hbind;
    [ destruct i; discriminate Hp | ].
  destruct vs as [ | v0 vs]; [ discriminate Hbind | ]. simpl in Hbind.
  inversion Hnodup as [ | p1 rest Hnotin Hnodup']. subst p1 rest.
  destruct i as [ | i]; simpl in Hp, Hv.
  - injection Hp as <-. injection Hv as <-.
    rewrite (lookup_bind_params_other params vs _ bound p0 Hbind Hnotin).
    apply lookup_head.
  - exact (IH i p v vs _ bound Hnodup' Hp Hv Hbind).
Qed.

(* ======================================================================== *)
(* An expression with parameters bound again                                *)
(* ======================================================================== *)

(* The kinds given to the [Let]s of the parameters are those that the
   handler gives the parameters. *)
Definition kinds_fit (kinds params_kinds : list flambda_kind) : Prop :=
  ∀ (i : nat) (kind : flambda_kind),
    nth_error kinds i = Some kind -> nth_error params_kinds i = Some kind.

Lemma kinds_fit_nil (params_kinds : list flambda_kind) :
  kinds_fit [] params_kinds.
Proof. intros i kind Hkind. destruct i; discriminate Hkind. Qed.

Lemma kinds_fit_refl (kinds : list flambda_kind) : kinds_fit kinds kinds.
Proof. intros i kind Hkind. exact Hkind. Qed.

Definition handler_kinds (h : continuation_handler) : list flambda_kind :=
  match h with
  | Handler _ metadata _ => hm_params_kinds metadata
  end.

(* The second expression is the first with, at some [Let_cont]s, [Let]s
   put at the start of the handler that bind parameters to what the
   fate of the continuation says they are always given, at the kinds
   that the handler gives them; the handlers give the same kinds. Every
   application fits the fate of its continuation, no [Apply] returns to
   a continuation of which something is said, and no binder binds a
   name that a fate in scope depends on. (The warning silenced is about
   induction schemes, which nothing uses.) *)
#[local] Set Warnings "-register-all".
Inductive aliased : fates -> expr -> expr -> Prop :=
  | Aliased_let (K : fates) (x : variable) (kind : flambda_kind) (d : named)
      (body body' : expr) :
    unfrozen K x ->
    aliased K body body' ->
    aliased K (Let x kind d body) (Let x kind d body')
  | Aliased_let_static (K : fates) (bindings : list static_binding)
      (body body' : expr) :
    aliased K body body' ->
    aliased K (Let_static bindings body) (Let_static bindings body')
  | Aliased_let_cont (K : fates) (k : continuation) (params : list variable)
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (kinds : list flambda_kind) (aliases : list (option simple))
      (body body' : expr) :
    aliased K handler handler' ->
    (∀ p : variable, p ∈ params -> unfrozen K p) ->
    aliases_fit aliases params ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    kinds_fit kinds (hm_params_kinds metadata) ->
    aliased ((k, (aliases, [])) :: K) body body' ->
    aliased K
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (Handler params metadata'
            (bind_aliases params kinds aliases handler'))
         body')
  | Aliased_let_cont_recursive (K : fates)
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    Forall2
      (fun kh kh' =>
         fst kh = fst kh'
         ∧ handler_params (snd kh) = handler_params (snd kh')
         ∧ handler_kinds (snd kh') = handler_kinds (snd kh)
         ∧ aliased (plain_fates handlers ++ K) (handler_expr (snd kh))
             (handler_expr (snd kh'))
         ∧ ∀ p : variable, p ∈ handler_params (snd kh) -> unfrozen K p)
      handlers handlers' ->
    handlers_bind_statics handlers' = handlers_bind_statics handlers ->
    aliased (plain_fates handlers ++ K) body body' ->
    aliased K (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | Aliased_loop (K : fates) (k : continuation) (params : list variable)
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (kinds : list flambda_kind) (aliases : list (option simple))
      (body body' : expr) :
    aliased ((k, (aliases, params)) :: K) handler handler' ->
    (∀ p : variable, p ∈ params -> unfrozen K p) ->
    aliases_fit aliases params ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    kinds_fit kinds (hm_params_kinds metadata) ->
    binds_statics handler' = binds_statics handler ->
    aliased ((k, (aliases, [])) :: K) body body' ->
    aliased K
      (Let_cont_recursive [(k, Handler params metadata handler)] body)
      (Let_cont_recursive
         [(k,
           Handler params metadata'
             (bind_aliases params kinds aliases handler'))]
         body')
  | Aliased_apply (K : fates) (ap : apply_expr) :
    no_alias (fate_of K (ap_continuation ap)) ->
    no_alias (fate_of K (ap_exn_continuation ap)) ->
    aliased K (Apply ap) (Apply ap)
  | Aliased_apply_cont (K : fates) (ac : apply_cont) :
    args_fit (fate_of K (ac_continuation ac)) (ac_args ac) ->
    aliased K (Apply_cont ac) (Apply_cont ac)
  | Aliased_switch (K : fates) (scrutinee : simple)
      (arms : list (Z * apply_cont)) :
    (∀ arm : Z * apply_cont,
      arm ∈ arms ->
      args_fit (fate_of K (ac_continuation (snd arm))) (ac_args (snd arm))) ->
    aliased K (Switch scrutinee arms) (Switch scrutinee arms)
  | Aliased_invalid (K : fates) : aliased K Invalid Invalid.

(* The handlers of a recursive [Let_cont] of which nothing is said, one
   with one: the same names, parameters and kinds, and expressions so
   related. *)
Definition handlers_aliased (inside K : fates)
    (handlers handlers' : list (continuation * continuation_handler)) : Prop :=
  Forall2
    (fun kh kh' =>
       fst kh = fst kh'
       ∧ handler_params (snd kh) = handler_params (snd kh')
       ∧ handler_kinds (snd kh') = handler_kinds (snd kh)
       ∧ aliased inside (handler_expr (snd kh)) (handler_expr (snd kh'))
       ∧ ∀ p : variable, p ∈ handler_params (snd kh) -> unfrozen K p)
    handlers handlers'.

Lemma handlers_aliased_lookup (inside K : fates) (k : continuation)
    (handlers handlers' : list (continuation * continuation_handler)) :
  handlers_aliased inside K handlers handlers' ->
  match lookup k handlers, lookup k handlers' with
  | Some h, Some h' =>
    handler_params h = handler_params h'
    ∧ handler_kinds h' = handler_kinds h
    ∧ aliased inside (handler_expr h) (handler_expr h')
    ∧ ∀ p : variable, p ∈ handler_params h -> unfrozen K p
  | None, None => True
  | _, _ => False
  end.
Proof.
  induction 1 as [ | [k1 h] [k1' h'] handlers handlers' (Hk & Hh) _ IH];
    simpl; [ exact I | ].
  simpl in Hk, Hh. subst k1'. destruct (Pos.eqb k k1); [ exact Hh | exact IH ].
Qed.

(* Every expression is related to itself, where nothing is said of any
   continuation. *)
Fixpoint aliased_refl (e : expr) :
  ∀ K : fates, (∀ k : continuation, no_alias (fate_of K k)) -> aliased K e e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros K Hno.
  - apply Aliased_let; [ | apply aliased_refl, Hno ].
    intros k. apply no_alias_unfrozen, Hno.
  - apply Aliased_let_static, aliased_refl, Hno.
  - pose proof (Aliased_let_cont K k params metadata metadata handler handler
      [] [] body body) as Hlet.
    rewrite bind_aliases_nil in Hlet. apply Hlet.
    + apply aliased_refl, Hno.
    + intros p _ k'. apply no_alias_unfrozen, Hno.
    + intros i s Hs. destruct i; discriminate Hs.
    + reflexivity.
    + apply kinds_fit_nil.
    + apply aliased_refl. intros k'.
      destruct (Pos.eq_dec k' k) as [-> | Hne];
        [ rewrite fate_of_head; apply no_alias_nil
        | rewrite fate_of_other by exact Hne; apply Hno ].
  - assert (Hno' :
      ∀ k : continuation, no_alias (fate_of (plain_fates handlers ++ K) k)).
    { intros k. rewrite fate_of_plain.
      destruct (lookup k handlers); [ apply no_alias_nil | apply Hno ]. }
    apply Aliased_let_cont_recursive;
      [ | reflexivity | apply aliased_refl, Hno' ].
    revert Hno'. generalize (plain_fates handlers ++ K). intros K' Hno'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    simpl. split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ apply aliased_refl, Hno' | ].
    intros p _ k'. apply no_alias_unfrozen, Hno.
  - apply Aliased_apply; apply Hno.
  - apply Aliased_apply_cont. intros i s Hs. exfalso. exact (Hno _ i s Hs).
  - apply Aliased_switch. intros arm _ i s Hs. exfalso. exact (Hno _ i s Hs).
  - apply Aliased_invalid.
Qed.

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

(* What the fate says a parameter is always given means, under the
   variables [vars], what it means under those of the closure; and
   inside the handler of a loop the parameter itself is bound to that. *)
Definition held (vars : env) (f : fate) (def_vars : env) : Prop :=
  ∀ (i : nat) (s : simple),
    alias_at f i s ->
    eval_simple vars s = eval_simple def_vars s
    ∧ ∀ p : variable,
        nth_error (snd f) i = Some p -> lookup p vars = eval_simple def_vars s.

(* What a continuation name stands for in the two runs, under the
   variables of the first, given the fate of the name: the same thing,
   if nothing is said of it; or closures whose handlers are related
   under the fates of where they were defined, the second with the
   [Let]s that the fate asks for, under variables bound alike and
   continuations in turn so related, the handlers defined with it being
   so related too; or the closures of a loop, alone in its [Let_cont],
   whose handler is related under its own fate with its parameters; or
   two that are related with nothing said, with the same extra
   arguments; or what is left of the same over-application, to return
   and to raise to two that are related with nothing said. *)
Inductive cont_aliased : env -> fate -> cont_value -> cont_value -> Prop :=
  | Same_aliased (vars : env) (f : fate) (c : cont_value) :
    no_alias f -> cont_aliased vars f c c
  | Closure_aliased (vars : env) (f : fate) (K : fates)
      (params : list variable) (metadata metadata' : handler_metadata)
      (handler handler' : expr) (kinds : list flambda_kind)
      (def_vars def_vars' : env) (def_conts def_conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    handlers_aliased (plain_fates recursive ++ K) K recursive recursive' ->
    aliased (plain_fates recursive ++ K) handler handler' ->
    (∀ p : variable, p ∈ params -> unfrozen K p) ->
    aliases_fit (fst f) params ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    kinds_fit kinds (hm_params_kinds metadata) ->
    (∀ x : variable, lookup x def_vars = lookup x def_vars') ->
    (∀ k : continuation,
      cont_aliased def_vars (fate_of K k) (resolve k def_conts)
        (resolve k def_conts')) ->
    held vars f def_vars ->
    cont_aliased vars f
      (Closure (Handler params metadata handler) def_vars def_conts recursive)
      (Closure
         (Handler params metadata'
            (bind_aliases params kinds (fst f) handler'))
         def_vars' def_conts' recursive')
  | Loop_aliased (vars : env) (f : fate) (K : fates) (k : continuation)
      (params : list variable) (metadata metadata' : handler_metadata)
      (handler handler' : expr) (kinds : list flambda_kind)
      (def_vars def_vars' : env) (def_conts def_conts' : cont_env) :
    aliased ((k, (fst f, params)) :: K) handler handler' ->
    (∀ p : variable, p ∈ params -> unfrozen K p) ->
    aliases_fit (fst f) params ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    kinds_fit kinds (hm_params_kinds metadata) ->
    (∀ x : variable, lookup x def_vars = lookup x def_vars') ->
    (∀ k' : continuation,
      cont_aliased def_vars (fate_of K k') (resolve k' def_conts)
        (resolve k' def_conts')) ->
    held vars f def_vars ->
    cont_aliased vars f
      (Closure (Handler params metadata handler) def_vars def_conts
         [(k, Handler params metadata handler)])
      (Closure
         (Handler params metadata'
            (bind_aliases params kinds (fst f) handler'))
         def_vars' def_conts'
         [(k,
           Handler params metadata'
             (bind_aliases params kinds (fst f) handler'))])
  | Extra_args_aliased (vars : env) (f : fate) (c c' : cont_value)
      (extra : list value) :
    no_alias f ->
    cont_aliased vars f c c' ->
    cont_aliased vars f (Extra_args c extra) (Extra_args c' extra)
  | Over_aliased (vars : env) (f f_ret f_exn : fate) (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    no_alias f ->
    no_alias f_ret ->
    no_alias f_exn ->
    cont_aliased vars f_ret return_to return_to' ->
    cont_aliased vars f_exn exn_to exn_to' ->
    cont_aliased vars f
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Inductive state_aliased : state -> state -> Prop :=
  | State_aliased (K : fates) (e e' : expr) (vars vars' : env)
      (conts conts' : cont_env) (statics : static_env) (hp : heap)
        (rd : read_record) :
    aliased K e e' ->
    (∀ x : variable, lookup x vars = lookup x vars') ->
    (∀ k : continuation,
      cont_aliased vars (fate_of K k) (resolve k conts) (resolve k conts')) ->
    state_aliased (State e vars conts statics hp rd)
      (State e' vars' conts' statics hp rd).

Lemma state_aliased_refl (s : state) : state_aliased s s.
Proof.
  destruct s as [e vars conts statics].
  apply (State_aliased []);
    [ apply aliased_refl; intros k; apply no_alias_nil | reflexivity
    | intros k; apply Same_aliased, no_alias_nil ].
Qed.

(* Other variables that agree with the first on the names that the fate
   depends on serve as well. *)
Lemma held_env (vars vars2 : env) (f : fate) (def_vars : env) :
  (∀ x : variable, frozen f x -> lookup x vars2 = lookup x vars) ->
  held vars f def_vars ->
  held vars2 f def_vars.
Proof.
  intros Hsame Hheld i s Halias. destruct (Hheld i s Halias) as [Hs Halt].
  split.
  - rewrite <- Hs. destruct s as [x | sym | c]; simpl; try reflexivity.
    apply Hsame. exists i, (Var x). split; [ exact Halias | left; reflexivity ].
  - intros p Hp. rewrite <- (Halt p Hp). apply Hsame. exists i, s.
    split; [ exact Halias | right; exact Hp ].
Qed.

Lemma cont_aliased_env (vars : env) (f : fate) (c c' : cont_value) :
  cont_aliased vars f c c' ->
  ∀ vars2 : env,
    (∀ x : variable, frozen f x -> lookup x vars2 = lookup x vars) ->
    cont_aliased vars2 f c c'.
Proof.
  induction 1
    as [ vars f c Hno
       | vars f K params metadata metadata' handler handler' kinds def_vars
           def_vars' def_conts def_conts' recursive recursive' Hrecursive
           Hhandler Hparams Hfit Hmeta Hkinds Hdef_vars Hdef_conts _ Hheld
       | vars f K k params metadata metadata' handler handler' kinds def_vars
           def_vars' def_conts def_conts' Hhandler Hparams Hfit Hmeta Hkinds
           Hdef_vars Hdef_conts _ Hheld
       | vars f c c' extra Hno Hc IHc
       | vars f f_ret f_exn remaining remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hno Hno_ret Hno_exn Hreturn IHreturn
           Hraise IHraise ];
    intros vars2 Hsame.
  - apply Same_aliased, Hno.
  - apply (Closure_aliased vars2 f K); try assumption.
    exact (held_env vars vars2 f def_vars Hsame Hheld).
  - apply (Loop_aliased vars2 f K); try assumption.
    exact (held_env vars vars2 f def_vars Hsame Hheld).
  - apply Extra_args_aliased; [ exact Hno | ]. apply IHc, Hsame.
  - apply (Over_aliased vars2 f f_ret f_exn); try assumption.
    + apply IHreturn. intros x Hx. exfalso.
      exact (no_alias_unfrozen _ x Hno_ret Hx).
    + apply IHraise. intros x Hx. exfalso.
      exact (no_alias_unfrozen _ x Hno_exn Hx).
Qed.

(* The closures of the handlers of a recursive [Let_cont] of which
   nothing is said, over continuations that are related. *)
Lemma rec_conts_aliased (K : fates)
    (handlers handlers' : list (continuation * continuation_handler))
    (vars def_vars def_vars' : env) (def_conts def_conts' : cont_env) :
  handlers_aliased (plain_fates handlers ++ K) K handlers handlers' ->
  (∀ x : variable, lookup x def_vars = lookup x def_vars') ->
  (∀ k : continuation,
    cont_aliased def_vars (fate_of K k) (resolve k def_conts)
      (resolve k def_conts')) ->
  (∀ k : continuation,
    cont_aliased vars (fate_of K k) (resolve k def_conts)
      (resolve k def_conts')) ->
  ∀ k : continuation,
    cont_aliased vars (fate_of (plain_fates handlers ++ K) k)
      (resolve k (rec_closures handlers def_vars def_conts ++ def_conts))
      (resolve k (rec_closures handlers' def_vars' def_conts' ++ def_conts')).
Proof.
  intros Hhandlers Hvars Hdef Hcur k.
  rewrite fate_of_plain, !resolve_rec_closures.
  pose proof (handlers_aliased_lookup _ _ k _ _ Hhandlers) as Hlookup.
  destruct (lookup k handlers) as [[params metadata handler] | ],
    (lookup k handlers') as [[params' metadata' handler'] | ];
    try contradiction; [ | apply Hcur ].
  destruct Hlookup as (Hparams & Hmeta & Hhandler & Hunfrozen).
  simpl in Hparams, Hmeta, Hhandler, Hunfrozen. subst params'.
  pose proof (Closure_aliased vars ([], []) K params metadata metadata'
    handler handler' [] def_vars def_vars' def_conts def_conts' handlers
    handlers' Hhandlers Hhandler Hunfrozen) as Hclosure.
  simpl in Hclosure. rewrite bind_aliases_nil in Hclosure.
  apply Hclosure;
    [ | exact Hmeta | apply kinds_fit_nil | exact Hvars | exact Hdef | ].
  - intros i s Hs. destruct i; discriminate Hs.
  - intros i s Hs. exfalso. exact (no_alias_nil [] i s Hs).
Qed.

(* ======================================================================== *)
(* Applying related continuations                                           *)
(* ======================================================================== *)

(* The values passed are what the fate says: at a position of which it
   says something, the value of that under the variables given, or of
   the parameter of the loop. *)
Definition given_values (vars : env) (f : fate) (vs : list value) : Prop :=
  ∀ (i : nat) (s : simple),
    alias_at f i s ->
    ∃ v : value,
      nth_error vs i = Some v
      ∧ (eval_simple vars s = Some v
         ∨ ∃ p : variable,
             nth_error (snd f) i = Some p ∧ lookup p vars = Some v).

(* Then, once the parameters are bound, each one of which the fate says
   something is bound to what that means under the variables of the
   closure, and under the new ones. *)
Lemma given_bound (vars : env) (f : fate) (params : list variable)
    (vs : list value) (def_vars bound : env) :
  aliases_fit (fst f) params ->
  held vars f def_vars ->
  given_values vars f vs ->
  bind_params params vs def_vars = Some bound ->
  ∀ (i : nat) (p : variable) (s : simple),
    nth_error params i = Some p ->
    alias_at f i s ->
    ∃ v : value,
      eval_simple def_vars s = Some v
      ∧ eval_simple bound s = Some v
      ∧ lookup p bound = Some v
      ∧ nth_error vs i = Some v.
Proof.
  intros Hfit Hheld Hgiven Hbind i p s Hp Hs.
  destruct (Hgiven i s Hs) as (v & Hv & Hval).
  destruct (Hheld i s Hs) as [Hdef Halt].
  destruct (Hfit i s Hs) as [Hnodup Hstable].
  assert (Hdef_v : eval_simple def_vars s = Some v).
  { destruct Hval as [Hval | (p0 & Hp0 & Hval)];
      [ rewrite <- Hdef; exact Hval | rewrite <- (Halt p0 Hp0); exact Hval ]. }
  exists v. split; [ exact Hdef_v | ]. split; [ | split; [ | exact Hv ] ].
  - rewrite <- Hdef_v. destruct s as [x | sym | c]; simpl; try reflexivity.
    apply (lookup_bind_params_other params vs def_vars bound x Hbind).
    exact (Hstable x eq_refl).
  - exact (bind_params_nth params i p v vs def_vars bound Hnodup Hp Hv Hbind).
Qed.

(* What two steps come to: the first is undefined; or both exit alike;
   or both step, the second then going on for some steps, to related
   states. *)
Definition result_aliased (result result' : step_result) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs0 hp rd => result' = Exit k vs0 hp rd
  | Step t =>
    ∃ (t1 : state) (n : nat) (t' : state),
      result' = Step t1 ∧ run n t1 = Step t' ∧ state_aliased t t'
  end.

(* The same full call, returning and raising to related continuations of
   which nothing is said. *)
Lemma call_code_aliased (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (f_ret f_exn : fate)
    (return_to return_to' exn_to exn_to' : cont_value) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  no_alias f_ret ->
  no_alias f_exn ->
  cont_aliased vars f_ret return_to return_to' ->
  cont_aliased vars f_exn exn_to exn_to' ->
  result_aliased
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hno_ret Hno_exn Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite Hundefined. exact I.
  - rewrite !Hcall. cbn [result_aliased]. eexists _, 0, _.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    apply (State_aliased
      [(code_return_continuation c, f_ret); (code_exn_continuation c, f_exn)]).
    + apply aliased_refl. intros k'. rewrite fate_of_pair.
      destruct (Pos.eqb k' (code_return_continuation c)); [ exact Hno_ret | ].
      destruct (Pos.eqb k' (code_exn_continuation c));
        [ exact Hno_exn | apply no_alias_nil ].
    + reflexivity.
    + intros k'. rewrite fate_of_pair, !resolve_pair.
      destruct (Pos.eqb k' (code_return_continuation c)).
      { apply (cont_aliased_env vars); [ exact Hreturn | ].
        intros y Hy. exfalso. exact (no_alias_unfrozen _ y Hno_ret Hy). }
      destruct (Pos.eqb k' (code_exn_continuation c));
        [ | apply Same_aliased, no_alias_nil ].
      apply (cont_aliased_env vars); [ exact Hraise | ].
      intros y Hy. exfalso. exact (no_alias_unfrozen _ y Hno_exn Hy).
Qed.

(* The same call of any shape, where the two return continuations take
   a partial application to related results. *)
Lemma run_call_aliased (shape : option call_shape) (return_arity : arity)
    (f_ret f_exn : fate) (return_to return_to' exn_to exn_to' : cont_value)
    (vars : env) (statics : static_env) (hp : heap) (rd : read_record) :
  no_alias f_ret ->
  no_alias f_exn ->
  cont_aliased vars f_ret return_to return_to' ->
  cont_aliased vars f_exn exn_to exn_to' ->
  (∀ partial : value,
    result_aliased (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  result_aliased
    (run_call
       (fun partial => apply_cont_value return_to [partial] statics hp rd)
       shape return_arity return_to exn_to statics hp rd)
    (run_call
       (fun partial => apply_cont_value return_to' [partial] statics hp rd)
       shape return_arity return_to' exn_to' statics hp rd).
Proof.
  intros Hno_ret Hno_exn Hreturn Hraise Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    cbn [run_call].
  - exact
      (call_code_aliased closure c vs args_arity return_arity f_ret f_exn _ _
         _ _ vars statics _ rd_ Hno_ret Hno_exn Hreturn Hraise).
  - apply
      (call_code_aliased closure c vs args_arity (code_result_arity c)
         ([], []) f_exn _ _ _ _ vars statics _ rd_ (no_alias_nil []) Hno_exn);
      [ | exact Hraise ].
    exact
      (Over_aliased vars ([], []) f_ret f_exn remaining remaining_arity
         return_arity return_to return_to' exn_to exn_to' (no_alias_nil [])
         Hno_ret Hno_exn Hreturn Hraise).
  - apply Hpartial.
  - exact I.
Qed.

(* A value among those given to a handler that has a step is of the
   kind that the [Let] of its position has. *)
Lemma given_kind (kinds params_kinds : list flambda_kind) (vs : list value)
    (i : nat) (v : value) (kind : flambda_kind) :
  kinds_fit kinds params_kinds ->
  kinds_eqb (map kind_of_value vs) params_kinds = true ->
  nth_error vs i = Some v ->
  nth_error kinds i = Some kind ->
  kind_of_value v = kind.
Proof.
  intros Hfit Hgiven Hv Hkind. apply kinds_eqb_eq in Hgiven.
  specialize (Hfit i kind Hkind). rewrite <- Hgiven in Hfit.
  rewrite (map_nth_error kind_of_value i vs Hv) in Hfit.
  injection Hfit as Hfit. exact Hfit.
Qed.

Lemma apply_cont_aliased (vars : env) (f : fate) (c c' : cont_value) :
  cont_aliased vars f c c' ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    given_values vars f vs ->
    match apply_cont_value c vs statics hp rd with
    | Undefined => True
    | Exit k vs0 hp' rd' => apply_cont_value c' vs statics hp rd
      = Exit k vs0 hp' rd'
    | Step t =>
      ∃ (t1 : state) (n : nat) (t' : state),
        apply_cont_value c' vs statics hp rd = Step t1
        ∧ run n t1 = Step t'
        ∧ state_aliased t t'
    end.
Proof.
  induction 1
    as [ vars f c Hno
       | vars f K params metadata metadata' handler handler' kinds def_vars
           def_vars' def_conts def_conts' recursive recursive' Hrecursive
           Hhandler Hparams Hfit Hmeta Hkinds Hdef_vars Hdef_conts _ Hheld
       | vars f K k params metadata metadata' handler handler' kinds def_vars
           def_vars' def_conts def_conts' Hhandler Hparams Hfit Hmeta Hkinds
           Hdef_vars Hdef_conts _ Hheld
       | vars f c c' extra Hno Hc IHc
       | vars f f_ret f_exn remaining remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hno Hno_ret Hno_exn Hreturn IHreturn
           Hraise _ ];
    intros vs statics hp rd Hgiven.
  - (* the same continuation *)
    destruct (apply_cont_value c vs statics _ rd_) as [t | k vs0 | ];
      [ | reflexivity | exact I ].
    exists t, 0, t. split; [ reflexivity | ].
    split; [ reflexivity | apply state_aliased_refl ].
  - (* related closures *)
    simpl. rewrite Hmeta.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Htested; [ | exact I ].
    destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct (bind_params_alike params vs def_vars def_vars' bound Hdef_vars
                Hbind)
      as (bound' & Hbind' & Hbound).
    rewrite Hbind'.
    destruct
      (run_bind_aliases params kinds (fst f) handler' bound'
         (rec_closures recursive' def_vars' def_conts' ++ def_conts') statics
           hp rd)
      as (n & vars' & Hrun & Hsame).
    { intros i p s Hp Hs.
      destruct (given_bound vars f params vs def_vars bound Hfit Hheld Hgiven
                  Hbind i p s Hp Hs)
        as (v & _ & Hs_bound & Hp_bound & Hv).
      exists v. split; [ | split ].
      - rewrite (eval_simple_agree bound bound' s);
          [ exact Hs_bound | intros x _; apply Hbound ].
      - rewrite <- Hbound. exact Hp_bound.
      - intros kind Hkind.
        exact (given_kind kinds _ vs i v kind Hkinds Htested Hv Hkind). }
    eexists _, n, _. split; [ reflexivity | ]. split; [ exact Hrun | ].
    apply (State_aliased (plain_fates recursive ++ K)); [ exact Hhandler | | ].
    + intros x. rewrite Hsame. apply Hbound.
    + apply rec_conts_aliased;
        [ exact Hrecursive | exact Hdef_vars | exact Hdef_conts | ].
      intros k. apply (cont_aliased_env def_vars); [ apply Hdef_conts | ].
      intros x Hx.
      apply (lookup_bind_params_other params vs def_vars bound x Hbind).
      intros Hin. exact (Hparams x Hin k Hx).
  - (* the closures of a loop *)
    simpl. rewrite Hmeta.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Htested; [ | exact I ].
    destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct (bind_params_alike params vs def_vars def_vars' bound Hdef_vars
                Hbind)
      as (bound' & Hbind' & Hbound).
    rewrite Hbind'.
    pose proof (given_bound vars f params vs def_vars bound Hfit Hheld Hgiven
      Hbind) as Hvalues.
    destruct
      (run_bind_aliases params kinds (fst f) handler' bound'
         ((k,
           Closure
             (Handler params metadata'
                (bind_aliases params kinds (fst f) handler'))
             def_vars' def_conts'
             [(k,
               Handler params metadata'
                 (bind_aliases params kinds (fst f) handler'))])
          :: def_conts')
         statics hp rd)
      as (n & vars' & Hrun & Hsame).
    { intros i p s Hp Hs.
      destruct (Hvalues i p s Hp Hs) as (v & _ & Hs_bound & Hp_bound & Hv).
      exists v. split; [ | split ].
      - rewrite (eval_simple_agree bound bound' s);
          [ exact Hs_bound | intros x _; apply Hbound ].
      - rewrite <- Hbound. exact Hp_bound.
      - intros kind Hkind.
        exact (given_kind kinds _ vs i v kind Hkinds Htested Hv Hkind). }
    eexists _, n, _. split; [ reflexivity | ]. split; [ exact Hrun | ].
    apply (State_aliased ((k, (fst f, params)) :: K)); [ exact Hhandler | | ].
    + intros x. rewrite Hsame. apply Hbound.
    + intros k1. destruct (Pos.eq_dec k1 k) as [-> | Hne].
      * rewrite fate_of_head. simpl. rewrite !resolve_head.
        apply (Loop_aliased bound (fst f, params) K k params metadata
          metadata' handler handler' kinds def_vars def_vars' def_conts
          def_conts');
          [ exact Hhandler | exact Hparams | exact Hfit | exact Hmeta
          | exact Hkinds | exact Hdef_vars | exact Hdef_conts | ].
        intros i s Hs. unfold alias_at in Hs. simpl in Hs.
        destruct (nth_error params i) as [p | ] eqn:Hp.
        -- destruct (Hvalues i p s Hp Hs)
             as (v & Hdef_v & Hs_bound & Hp_bound & _).
           split; [ rewrite Hdef_v; exact Hs_bound | ].
           intros p0 Hp0. simpl in Hp0. rewrite Hp in Hp0.
           injection Hp0 as <-. rewrite Hdef_v. exact Hp_bound.
        -- split; [ | intros p0 Hp0; simpl in Hp0; congruence ].
           destruct (Hfit i s Hs) as [_ Hstable].
           destruct s as [x | sym | c]; simpl; try reflexivity.
           apply (lookup_bind_params_other params vs def_vars bound x Hbind).
           exact (Hstable x eq_refl).
      * rewrite fate_of_other by exact Hne. simpl.
        rewrite !resolve_other by exact Hne.
        apply (cont_aliased_env def_vars); [ apply Hdef_conts | ].
        intros x Hx.
        apply (lookup_bind_params_other params vs def_vars bound x Hbind).
        intros Hin. exact (Hparams x Hin k1 Hx).
  - (* the same extra arguments *)
    simpl. apply IHc. intros i s Hs. exfalso. exact (Hno i s Hs).
  - (* what is left of the same over-application *)
    rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    unfold call_function.
    apply
      (run_call_aliased
         (shape_of_call statics v Indirect_unknown_arity remaining
            remaining_arity return_arity)
         return_arity f_ret f_exn return_to return_to' exn_to exn_to' vars
         statics _ rd_ Hno_ret Hno_exn Hreturn Hraise).
    intros partial. apply IHreturn. intros i s Hs. exfalso.
    exact (Hno_ret i s Hs).
Qed.

(* The same call, returning and raising to related continuations of
   which nothing is said. *)
Lemma call_function_aliased (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) (f_ret f_exn : fate)
    (return_to return_to' exn_to exn_to' : cont_value) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  no_alias f_ret ->
  no_alias f_exn ->
  cont_aliased vars f_ret return_to return_to' ->
  cont_aliased vars f_exn exn_to exn_to' ->
  result_aliased
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hno_ret Hno_exn Hreturn Hraise. unfold call_function.
  apply
    (run_call_aliased
       (shape_of_call statics callee kind vs args_arity return_arity)
       return_arity f_ret f_exn return_to return_to' exn_to exn_to' vars
       statics _ rd_ Hno_ret Hno_exn Hreturn Hraise).
  intros partial. apply (apply_cont_aliased vars f_ret _ _ Hreturn).
  intros i s Hs. exfalso. exact (Hno_ret i s Hs).
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Definition same_values (last : state) (vs vs' : list value) : Prop := vs' = vs.

Lemma answered_one (t t' s' : state) :
  step s' = Step t' ->
  state_aliased t t' ->
  answered state_aliased same_values (Step t) s'.
Proof.
  intros Hstep Hrel. exists 0, t'.
  split; [ rewrite run_succ, Hstep; reflexivity | exact Hrel ].
Qed.

Lemma aliased_step (st st' : state) :
  state_aliased st st' -> answered state_aliased same_values (step st) st'.
Proof.
  intros [K e e' vars vars' conts conts' statics hp rd Haliased Hvars Hconts].
  destruct Haliased
    as [ K x kind d body body' Hx Hbody | K bindings body body' Hbody
       | K k params metadata metadata' handler handler' kinds aliases body
           body' Hhandler Hparams Hfit Hmeta Hkinds Hbody
       | K handlers handlers' body body' Hhandlers Hbinds Hbody
       | K k params metadata metadata' handler handler' kinds aliases body
           body' Hhandler Hparams Hfit Hmeta Hkinds Hbinds Hbody
       | K ap Hret Hexn | K ac Hargs | K scrutinee arms Harms | K ].
  - (* Let *)
    rewrite step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | exact I ].
    apply
      (answered_one _ (State body' ((x, v) :: vars') conts' statics hp' rd')).
    + rewrite step_let_exec, (exec_named_agree statics hp rd vars vars' d), Hd,
        Hkind
        by (intros y _; apply Hvars).
      reflexivity.
    + apply (State_aliased K); [ exact Hbody | | ].
      * intros y. simpl.
        destruct (Pos.eqb_spec y x); [ reflexivity | apply Hvars ].
      * intros k. apply (cont_aliased_env vars); [ apply Hconts | ].
        intros y Hy. simpl.
        destruct (Pos.eqb_spec y x) as [-> | _]; [ | reflexivity ].
        exfalso. exact (Hx k Hy).
  - (* Let_static *)
    rewrite step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hextended; [ | exact I ].
    apply (answered_one _ (State body' vars' conts' extended hp rd)).
    + rewrite step_let_static,
        (eval_static_bindings_agree vars vars' bindings), Hextended
        by (intros y _; apply Hvars).
      reflexivity.
    + apply (State_aliased K); assumption.
  - (* Let_cont_non_recursive *)
    rewrite step_let_cont_non_recursive.
    apply (answered_one _
      (State body' vars'
         ((k,
           Closure
             (Handler params metadata'
                (bind_aliases params kinds aliases handler'))
             vars' conts' [])
          :: conts')
         statics hp rd)).
    + apply step_let_cont_non_recursive.
    + apply (State_aliased ((k, (aliases, [])) :: K));
        [ exact Hbody | exact Hvars | ].
      intros k'. destruct (Pos.eq_dec k' k) as [-> | Hne].
      * rewrite fate_of_head, !resolve_head.
        apply (Closure_aliased vars (aliases, []) K params metadata metadata'
          handler handler' kinds vars vars' conts conts' [] []);
          [ apply Forall2_nil | exact Hhandler | exact Hparams | exact Hfit
          | exact Hmeta | exact Hkinds | exact Hvars | exact Hconts | ].
        intros i s Hs. split; [ reflexivity | ].
        intros p Hp. simpl in Hp. destruct i; discriminate Hp.
      * rewrite fate_of_other, !resolve_other by exact Hne. apply Hconts.
  - (* Let_cont_recursive, nothing said of its continuations *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hstatic; [ exact I | ].
    apply (answered_one _
      (State body' vars' (rec_closures handlers' vars' conts' ++ conts')
         statics hp rd)).
    + rewrite step_let_cont_recursive, Hbinds. reflexivity.
    + apply (State_aliased (plain_fates handlers ++ K));
        [ exact Hbody | exact Hvars | ].
      apply rec_conts_aliased; assumption.
  - (* a loop *)
    rewrite step_let_cont_recursive. unfold handlers_bind_statics. simpl.
    destruct (binds_statics handler) eqn:Hstatic; [ exact I | ]. simpl.
    apply (answered_one _
      (State body' vars'
         ((k,
           Closure
             (Handler params metadata'
                (bind_aliases params kinds aliases handler'))
             vars' conts'
             [(k,
               Handler params metadata'
                 (bind_aliases params kinds aliases handler'))])
          :: conts')
         statics hp rd)).
    + rewrite step_let_cont_recursive. unfold handlers_bind_statics. simpl.
      rewrite bind_aliases_binds_statics, Hbinds. reflexivity.
    + apply (State_aliased ((k, (aliases, [])) :: K));
        [ exact Hbody | exact Hvars | ].
      intros k'. destruct (Pos.eq_dec k' k) as [-> | Hne].
      * rewrite fate_of_head, !resolve_head.
        apply (Loop_aliased vars (aliases, []) K k params metadata metadata'
          handler handler' kinds vars vars' conts conts');
          [ exact Hhandler | exact Hparams | exact Hfit | exact Hmeta
          | exact Hkinds | exact Hvars | exact Hconts | ].
        intros i s Hs. split; [ reflexivity | ].
        intros p Hp. simpl in Hp. destruct i; discriminate Hp.
      * rewrite fate_of_other, !resolve_other by exact Hne. apply Hconts.
  - (* Apply: the same call, returning to related continuations *)
    rewrite step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra; [ | exact I ].
    assert (Hraise :
      cont_aliased vars (fate_of K (ap_exn_continuation ap))
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        (with_extra_args (resolve (ap_exn_continuation ap) conts') extra)).
    { destruct extra as [ | v extra]; simpl; [ apply Hconts | ].
      apply Extra_args_aliased; [ exact Hexn | apply Hconts ]. }
    pose proof
      (call_function_aliased callee (ap_call_kind ap) vs (ap_args_arity ap)
         (ap_return_arity ap) _ _ _ _ _ _ vars statics hp rd Hret Hexn
         (Hconts (ap_continuation ap)) Hraise)
      as Hcall.
    assert (Hstep' :
      step (State (Apply ap) vars' conts' statics hp rd)
      = call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
          (ap_return_arity ap) (resolve (ap_continuation ap) conts')
          (with_extra_args (resolve (ap_exn_continuation ap) conts') extra)
          statics hp rd).
    { rewrite step_apply, (eval_simple_agree vars vars' (ap_callee ap)),
        Hcallee, (eval_simples_agree vars vars' (ap_args ap)), Hvs,
        (eval_simples_agree vars vars' (exn_extra_simples ap)), Hextra
        by (intros y _; apply Hvars).
      reflexivity. }
    destruct
      (call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
         (ap_return_arity ap) (resolve (ap_continuation ap) conts)
         (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
         statics hp rd)
      as [t | k vs0 | ]; cbn [result_aliased] in Hcall; [ | | exact I ].
    + destruct Hcall as (t1 & n & t' & Hfirst & Hrun & Hrel).
      exists n, t'. split; [ | exact Hrel ].
      rewrite run_succ, Hstep', Hfirst. exact Hrun.
    + exists vs0. split; [ rewrite Hstep'; exact Hcall | reflexivity ].
  - (* Apply_cont *)
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    assert (Hgiven : given_values vars (fate_of K (ac_continuation ac)) vs).
    { intros i s Hs. destruct (Hargs i s Hs) as (arg & Harg & Hfits).
      destruct (eval_simples_nth vars (ac_args ac) vs i arg Hvs Harg)
        as (v & Hv & Heval).
      exists v. split; [ exact Hv | ].
      destruct Hfits as [-> | (p & Hp & ->)];
        [ left; exact Heval
        | right; exists p; split; [ exact Hp | exact Heval ] ]. }
    pose proof (apply_cont_aliased vars _ _ _ (Hconts (ac_continuation ac)) vs
      statics hp rd Hgiven) as Happly.
    assert (Hstep' :
      step (State (Apply_cont ac) vars' conts' statics hp rd)
      = apply_cont_value (resolve (ac_continuation ac) conts') vs statics hp
        rd).
    { rewrite step_apply_cont, (eval_simples_agree vars vars' (ac_args ac)),
        Hvs by (intros y _; apply Hvars).
      reflexivity. }
    destruct
      (apply_cont_value (resolve (ac_continuation ac) conts) vs statics hp rd)
      as [t | k vs0 | ]; [ | | exact I ].
    + destruct Happly as (t1 & n & t' & Hfirst & Hrun & Hrel).
      exists n, t'. split; [ | exact Hrel ].
      rewrite run_succ, Hstep', Hfirst. exact Hrun.
    + exists vs0. split; [ rewrite Hstep'; exact Happly | reflexivity ].
  - (* Switch *)
    rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | exact I ].
    apply (answered_one _ (State (Apply_cont ac) vars' conts' statics hp rd)).
    + rewrite step_switch, (eval_simple_agree vars vars' scrutinee), Hv, Hi,
        Hfind by (intros y _; apply Hvars).
      reflexivity.
    + apply (State_aliased K); [ | exact Hvars | exact Hconts ].
      apply Aliased_apply_cont.
      exact (Harms (i, ac) (find_arm_in i arms ac Hfind)).
  - (* Invalid *)
    rewrite step_invalid. exact I.
Qed.

(* ======================================================================== *)
(* The pass relates a term to its result                                    *)
(* ======================================================================== *)

Fixpoint bind_aliased_params_binds_statics (e : expr) :
  binds_statics (bind_aliased_params e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply bind_aliased_params_binds_statics.
  - rewrite bind_aliased_params_binds_statics. reflexivity.
  - rewrite bind_aliases_binds_statics, !bind_aliased_params_binds_statics.
    reflexivity.
  - rewrite bind_aliased_params_binds_statics. f_equal.
    generalize (alone handlers) (Let_cont_recursive handlers body).
    intros flag whole.
    induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
      [ reflexivity | ].
    rewrite bind_aliases_binds_statics, bind_aliased_params_binds_statics, IH.
    reflexivity.
Qed.

Lemma simple_eqb_eq (s s' : simple) : simple_eqb s s' = true -> s = s'.
Proof.
  unfold simple_eqb. destruct (simple_eq_dec s s'); [ auto | discriminate ].
Qed.

(* What an application may pass at a position of which the fate says
   something. *)
Definition accepts (f : fate) (i : nat) (s : simple) (arg : simple) : bool :=
  simple_eqb arg s
  || match nth_error (snd f) i with
     | Some p => simple_eqb arg (Var p)
     | None => false
     end.

Lemma given_fits (k : continuation) (f : fate) (i : nat) (s : simple)
    (ac : apply_cont) :
  given k i (accepts f i s) ac = true ->
  ac_continuation ac = k ->
  ∃ arg : simple,
    nth_error (ac_args ac) i = Some arg
    ∧ (arg = s ∨ ∃ p : variable, nth_error (snd f) i = Some p ∧ arg = Var p).
Proof.
  unfold given, accepts. intros Hgiven Hk.
  rewrite Hk, Pos.eqb_refl in Hgiven. simpl in Hgiven.
  destruct (nth_error (ac_args ac) i) as [arg | ]; [ | discriminate ].
  exists arg. split; [ reflexivity | ].
  apply orb_true_iff in Hgiven as [Heq | Halt];
    [ left; apply simple_eqb_eq, Heq | right ].
  destruct (nth_error (snd f) i) as [p | ]; [ | discriminate ].
  exists p. split; [ reflexivity | apply simple_eqb_eq, Halt ].
Qed.

Lemma given_weaken (k : continuation) (i : nat) (ok ok' : simple -> bool)
    (ac : apply_cont) :
  (∀ arg : simple, ok arg = true -> ok' arg = true) ->
  given k i ok ac = true ->
  given k i ok' ac = true.
Proof.
  unfold given. intros Hok.
  destruct (negb (Pos.eqb (ac_continuation ac) k)); simpl; [ auto | ].
  destruct (nth_error (ac_args ac) i); [ apply Hok | auto ].
Qed.

Fixpoint always_given_weaken (k : continuation) (i : nat)
    (ok ok' : simple -> bool) (e : expr) :
  (∀ arg : simple, ok arg = true -> ok' arg = true) ->
  always_given k i ok e = true ->
  always_given k i ok' e = true.
Proof.
  intros Hok.
  destruct e
    as [ x kind d body | bindings body | k0 [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl.
  - apply always_given_weaken, Hok.
  - apply always_given_weaken, Hok.
  - intros Hgiven. apply andb_true_iff in Hgiven as [Hhandler Hbody].
    apply andb_true_iff.
    split;
      [ exact (always_given_weaken k i ok ok' handler Hok Hhandler)
      | exact (always_given_weaken k i ok ok' body Hok Hbody) ].
  - intros Hgiven. apply andb_true_iff in Hgiven as [Hhandlers Hbody].
    apply andb_true_iff.
    split; [ | exact (always_given_weaken k i ok ok' body Hok Hbody) ].
    clear Hbody.
    induction handlers as [ | [k1 [params metadata handler]] handlers IH];
      [ reflexivity | ].
    simpl in Hhandlers |- *.
    apply andb_true_iff in Hhandlers as [Hhandler Hrest].
    apply andb_true_iff.
    split;
      [ exact (always_given_weaken k i ok ok' handler Hok Hhandler)
      | exact (IH Hrest) ].
  - auto.
  - apply given_weaken, Hok.
  - intros Hgiven.
    induction arms as [ | [discriminant ac] arms IH]; [ reflexivity | ].
    simpl in Hgiven |- *. apply andb_true_iff in Hgiven as [Harm Hrest].
    apply andb_true_iff.
    split; [ exact (given_weaken k i ok ok' ac Hok Harm) | exact (IH Hrest) ].
  - auto.
Qed.

Lemma existsb_false_in {A : Type} (f : A -> bool) (l : list A) (a : A) :
  existsb f l = false -> a ∈ l -> f a = false.
Proof.
  intros Hfalse Hin. destruct (f a) eqn:Hfa; [ | reflexivity ].
  rewrite <- Hfalse. symmetry. apply existsb_exists. exists a.
  split; assumption.
Qed.

(* What the relation asks of an expression under the fates [K], as the
   pass can check it: every application of a continuation fits its fate;
   the expression binds no name that a fate depends on; and no [Apply]
   returns to a continuation of which something is said. *)
Definition safe (K : fates) (e : expr) : Prop :=
  ∀ k : continuation,
    (∀ (i : nat) (s : simple),
      alias_at (fate_of K k) i s ->
      always_given k i (accepts (fate_of K k) i s) e = true)
    ∧ (∀ x : variable, frozen (fate_of K k) x -> binds_var x e = false)
    ∧ (∀ (i : nat) (s : simple),
        alias_at (fate_of K k) i s -> returned_to k e = false).

Lemma safe_nil (e : expr) : safe [] e.
Proof.
  intros k. split; [ | split ].
  - intros i s Hs. exfalso. exact (no_alias_nil [] i s Hs).
  - intros x Hx. exfalso. exact (no_alias_unfrozen _ x (no_alias_nil []) Hx).
  - intros i s Hs. exfalso. exact (no_alias_nil [] i s Hs).
Qed.

Lemma safe_let (K : fates) (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  safe K (Let x kind d body) -> safe K body ∧ unfrozen K x.
Proof.
  intros Hsafe. split.
  - intros k. destruct (Hsafe k) as (Hgiven & Hbinds & Hret).
    split; [ exact Hgiven | ]. split; [ | exact Hret ].
    intros y Hy. specialize (Hbinds y Hy). simpl in Hbinds.
    apply orb_false_iff in Hbinds. apply Hbinds.
  - intros k Hx. destruct (Hsafe k) as (_ & Hbinds & _).
    specialize (Hbinds x Hx). simpl in Hbinds.
    rewrite Pos.eqb_refl in Hbinds. discriminate Hbinds.
Qed.

Lemma safe_let_cont (K : fates) (k0 : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr) :
  safe K (Let_cont_non_recursive k0 (Handler params metadata handler) body) ->
  safe K handler ∧ safe K body ∧ ∀ p : variable, p ∈ params -> unfrozen K p.
Proof.
  intros Hsafe. split; [ | split ].
  - intros k. destruct (Hsafe k) as (Hgiven & Hbinds & Hret).
    split; [ | split ].
    + intros i s Hs. specialize (Hgiven i s Hs). simpl in Hgiven.
      apply andb_true_iff in Hgiven. apply Hgiven.
    + intros x Hx. specialize (Hbinds x Hx). simpl in Hbinds.
      apply orb_false_iff in Hbinds as [Hhandler _].
      apply orb_false_iff in Hhandler. apply Hhandler.
    + intros i s Hs. specialize (Hret i s Hs). simpl in Hret.
      apply orb_false_iff in Hret. apply Hret.
  - intros k. destruct (Hsafe k) as (Hgiven & Hbinds & Hret).
    split; [ | split ].
    + intros i s Hs. specialize (Hgiven i s Hs). simpl in Hgiven.
      apply andb_true_iff in Hgiven. apply Hgiven.
    + intros x Hx. specialize (Hbinds x Hx). simpl in Hbinds.
      apply orb_false_iff in Hbinds. apply Hbinds.
    + intros i s Hs. specialize (Hret i s Hs). simpl in Hret.
      apply orb_false_iff in Hret. apply Hret.
  - intros p Hin k Hp. destruct (Hsafe k) as (_ & Hbinds & _).
    specialize (Hbinds p Hp). simpl in Hbinds.
    apply orb_false_iff in Hbinds as [Hhandler _].
    apply orb_false_iff in Hhandler as [Hamong _].
    apply among_In in Hin. congruence.
Qed.

Lemma safe_let_cont_recursive (K : fates)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  safe K (Let_cont_recursive handlers body) ->
  safe K body
  ∧ ∀ (k0 : continuation) (params : list variable)
      (metadata : handler_metadata) (handler : expr),
      (k0, Handler params metadata handler) ∈ handlers ->
      safe K handler ∧ ∀ p : variable, p ∈ params -> unfrozen K p.
Proof.
  intros Hsafe. split.
  - intros k. destruct (Hsafe k) as (Hgiven & Hbinds & Hret).
    split; [ | split ].
    + intros i s Hs. specialize (Hgiven i s Hs). simpl in Hgiven.
      apply andb_true_iff in Hgiven. apply Hgiven.
    + intros x Hx. specialize (Hbinds x Hx). simpl in Hbinds.
      apply orb_false_iff in Hbinds. apply Hbinds.
    + intros i s Hs. specialize (Hret i s Hs). simpl in Hret.
      apply orb_false_iff in Hret. apply Hret.
  - intros k0 params metadata handler Hin.
    assert (Hbinds_handler :
      ∀ (k : continuation) (x : variable),
        frozen (fate_of K k) x ->
        among x params = false ∧ binds_var x handler = false).
    { intros k x Hx. destruct (Hsafe k) as (_ & Hbinds & _).
      specialize (Hbinds x Hx). simpl in Hbinds.
      apply orb_false_iff in Hbinds as [Hhandlers _].
      pose proof (existsb_false_in _ _ _ Hhandlers Hin) as Hhandler.
      simpl in Hhandler. apply orb_false_iff in Hhandler. exact Hhandler. }
    split.
    + intros k. destruct (Hsafe k) as (Hgiven & _ & Hret).
      split; [ | split ].
      * intros i s Hs. specialize (Hgiven i s Hs). simpl in Hgiven.
        apply andb_true_iff in Hgiven as [Hhandlers _].
        rewrite forallb_forall in Hhandlers.
        exact (Hhandlers _ Hin).
      * intros x Hx. exact (proj2 (Hbinds_handler k x Hx)).
      * intros i s Hs. specialize (Hret i s Hs). simpl in Hret.
        apply orb_false_iff in Hret as [Hhandlers _].
        exact (existsb_false_in _ _ _ Hhandlers Hin).
    + intros p Hp k Hfrozen.
      destruct (Hbinds_handler k p Hfrozen) as [Hamong _].
      apply among_In in Hp. congruence.
Qed.

Lemma safe_cons (K : fates) (k0 : continuation) (f : fate) (e : expr) :
  safe K e ->
  (∀ (i : nat) (s : simple),
    alias_at f i s -> always_given k0 i (accepts f i s) e = true) ->
  (∀ x : variable, frozen f x -> binds_var x e = false) ->
  (∀ (i : nat) (s : simple), alias_at f i s -> returned_to k0 e = false) ->
  safe ((k0, f) :: K) e.
Proof.
  intros Hsafe Hgiven Hbinds Hret k.
  destruct (Pos.eq_dec k k0) as [-> | Hne];
    [ rewrite fate_of_head; auto
    | rewrite fate_of_other by exact Hne; apply Hsafe ].
Qed.

Lemma safe_plain (K : fates)
    (handlers : list (continuation * continuation_handler)) (e : expr) :
  safe K e -> safe (plain_fates handlers ++ K) e.
Proof.
  intros Hsafe k. rewrite fate_of_plain.
  destruct (lookup k handlers); [ | apply Hsafe ].
  split; [ | split ].
  - intros i s Hs. exfalso. exact (no_alias_nil [] i s Hs).
  - intros x Hx. exfalso. exact (no_alias_unfrozen _ x (no_alias_nil []) Hx).
  - intros i s Hs. exfalso. exact (no_alias_nil [] i s Hs).
Qed.

(* What [aliases_of] answers of a position. *)
Lemma nth_error_aliases_from (params : list variable) :
  ∀ (j i : nat) (alias_at : nat -> variable -> option simple),
    nth_error (aliases_from j params alias_at) i
    = match nth_error params i with
      | Some param => Some (alias_at (j + i) param)
      | None => None
      end.
Proof.
  induction params as [ | param params IH]; intros j i alias_at;
    [ destruct i; reflexivity | ].
  destruct i as [ | i]; simpl; [ rewrite Nat.add_0_r; reflexivity | ].
  rewrite IH. replace (S j + i) with (j + S i) by lia. reflexivity.
Qed.

Lemma aliases_of_alias (k : continuation) (params : list variable)
    (metadata : handler_metadata) (loop : option expr) (body whole : expr)
    (i : nat) (s : simple) :
  nth_error (aliases_of k params metadata loop body whole) i = Some (Some s) ->
  returned_to k whole = false
  ∧ NoDup params
  ∧ stable s whole = true
  ∧ always_given k i (fun arg => simple_eqb arg s) body = true
  ∧ ∃ param : variable,
      nth_error params i = Some param
      ∧ match loop with
        | None => True
        | Some handler =>
          always_given k i
            (fun arg => simple_eqb arg s || simple_eqb arg (Var param))
            handler
          = true
          ∧ binds_var param handler = false
        end.
Proof.
  unfold aliases_of.
  destruct (returned_to k whole) eqn:Hret; simpl;
    [ destruct i; discriminate | ].
  destruct (hm_is_exn_handler metadata); simpl;
    [ destruct i; discriminate | ].
  destruct (distinct params) eqn:Hdistinct; simpl;
    [ | destruct i; discriminate ].
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    simpl; [ | destruct i; discriminate ].
  destruct (some_args k body) as [args | ]; [ | destruct i; discriminate ].
  rewrite nth_error_aliases_from. simpl.
  destruct (nth_error params i) as [param | ] eqn:Hparam; [ | discriminate ].
  destruct (nth_error args i) as [s0 | ]; [ | discriminate ].
  match goal with
  | |- Some (if ?check then _ else _) = _ -> _ =>
    destruct check eqn:Hcheck; [ | discriminate ]
  end.
  intros [= <-].
  apply andb_true_iff in Hcheck as [Hcheck Hloop].
  apply andb_true_iff in Hcheck as [Hstable Hgiven].
  split; [ reflexivity | ]. split; [ apply distinct_NoDup, Hdistinct | ].
  split; [ exact Hstable | ]. split; [ exact Hgiven | ].
  exists param. split; [ reflexivity | ].
  destruct loop as [handler | ]; [ | exact I ].
  apply andb_true_iff in Hloop as [Hhandler Hbinds].
  split; [ exact Hhandler | apply negb_true_iff, Hbinds ].
Qed.

(* A variable that is stable in a [Let_cont] is not bound by its parts. *)
Lemma stable_var_let_cont (x : variable) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  stable (Var x)
    (Let_cont_non_recursive k (Handler params metadata handler) body)
  = true ->
  among x params = false ∧ binds_var x handler = false
  ∧ binds_var x body = false.
Proof.
  simpl. intros Hstable. apply negb_true_iff in Hstable.
  apply orb_false_iff in Hstable as [Hhandler Hbody].
  apply orb_false_iff in Hhandler as [Hamong Hhandler]. auto.
Qed.

Lemma stable_var_loop (x : variable) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  stable (Var x)
    (Let_cont_recursive [(k, Handler params metadata handler)] body)
  = true ->
  among x params = false ∧ binds_var x handler = false
  ∧ binds_var x body = false.
Proof.
  simpl. intros Hstable. apply negb_true_iff in Hstable.
  apply orb_false_iff in Hstable as [Hhandlers Hbody].
  apply orb_false_iff in Hhandlers as [Hhandler _].
  apply orb_false_iff in Hhandler as [Hamong Hhandler]. auto.
Qed.

Fixpoint bind_aliased_params_aliased (e : expr) :
  ∀ K : fates, safe K e -> aliased K e (bind_aliased_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros K Hsafe.
  - destruct (safe_let K x kind d body Hsafe) as [Hbody Hx].
    apply Aliased_let; [ exact Hx | apply bind_aliased_params_aliased, Hbody ].
  - apply Aliased_let_static, bind_aliased_params_aliased. exact Hsafe.
  - destruct (safe_let_cont K k params metadata handler body Hsafe)
      as (Hhandler & Hbody & Hparams).
    set (whole :=
      Let_cont_non_recursive k (Handler params metadata handler) body).
    set (aliases := aliases_of k params metadata None body whole).
    assert (Halias :
      ∀ (i : nat) (s : simple),
        nth_error aliases i = Some (Some s) ->
        returned_to k body = false
        ∧ NoDup params
        ∧ (∀ x : variable,
            s = Var x -> x ∉ params ∧ binds_var x body = false)
        ∧ always_given k i (fun arg => simple_eqb arg s) body = true).
    { intros i s Hs.
      destruct (aliases_of_alias k params metadata None body whole i s Hs)
        as (Hret & Hnodup & Hstable & Hgiven & _).
      simpl in Hret. apply orb_false_iff in Hret as [_ Hret].
      split; [ exact Hret | ]. split; [ exact Hnodup | ].
      split; [ | exact Hgiven ].
      intros x ->.
      destruct (stable_var_let_cont x k params metadata handler body Hstable)
        as (Hamong & _ & Hbinds).
      split; [ | exact Hbinds ].
      intros Hin. apply among_In in Hin. congruence. }
    apply (Aliased_let_cont K k params metadata metadata handler
      (bind_aliased_params handler) (hm_params_kinds metadata) aliases body
      (bind_aliased_params body)).
    + apply bind_aliased_params_aliased, Hhandler.
    + exact Hparams.
    + intros i s Hs. destruct (Halias i s Hs) as (_ & Hnodup & Hstable & _).
      split; [ exact Hnodup | ]. intros x Hx. exact (proj1 (Hstable x Hx)).
    + reflexivity.
    + apply kinds_fit_refl.
    + apply bind_aliased_params_aliased, safe_cons; [ exact Hbody | | | ].
      * intros i s Hs. destruct (Halias i s Hs) as (_ & _ & _ & Hgiven).
        apply (always_given_weaken k i (fun arg => simple_eqb arg s));
          [ | exact Hgiven ].
        intros arg Harg. unfold accepts. rewrite Harg. reflexivity.
      * intros x (i & s & Hs & [-> | Halt]);
          [ | simpl in Halt; destruct i; discriminate Halt ].
        destruct (Halias i (Var x) Hs) as (_ & _ & Hstable & _).
        exact (proj2 (Hstable x eq_refl)).
      * intros i s Hs. exact (proj1 (Halias i s Hs)).
  - destruct (safe_let_cont_recursive K handlers body Hsafe)
      as [Hbody Hhandlers].
    destruct (alone handlers) eqn:Halone.
    + (* a loop *)
      destruct handlers as [ | [k [params metadata handler]] [ | kh rest]];
        try discriminate Halone.
      clear Halone.
      destruct (Hhandlers k params metadata handler (or_introl eq_refl))
        as [Hhandler Hparams].
      set (whole :=
        Let_cont_recursive [(k, Handler params metadata handler)] body).
      set (aliases := aliases_of k params metadata (Some handler) body whole).
      assert (Halias :
        ∀ (i : nat) (s : simple),
          nth_error aliases i = Some (Some s) ->
          returned_to k handler = false ∧ returned_to k body = false
          ∧ NoDup params
          ∧ (∀ x : variable,
              s = Var x ->
              x ∉ params ∧ binds_var x handler = false
              ∧ binds_var x body = false)
          ∧ always_given k i (fun arg => simple_eqb arg s) body = true
          ∧ ∃ param : variable,
              nth_error params i = Some param
              ∧ always_given k i
                  (fun arg => simple_eqb arg s || simple_eqb arg (Var param))
                  handler
                = true
              ∧ binds_var param handler = false).
      { intros i s Hs.
        destruct
          (aliases_of_alias k params metadata (Some handler) body whole i s Hs)
          as (Hret & Hnodup & Hstable & Hgiven & param & Hparam & Hloop
              & Hbinds).
        simpl in Hret. apply orb_false_iff in Hret as [Hret_handler Hret_body].
        apply orb_false_iff in Hret_handler as [Hret_handler _].
        split; [ exact Hret_handler | ]. split; [ exact Hret_body | ].
        split; [ exact Hnodup | ]. split; [ | split; [ exact Hgiven | ] ].
        - intros x ->.
          destruct (stable_var_loop x k params metadata handler body Hstable)
            as (Hamong & Hbinds_handler & Hbinds_body).
          split; [ | split; assumption ].
          intros Hin. apply among_In in Hin. congruence.
        - exists param. split; [ exact Hparam | ].
          split; [ exact Hloop | exact Hbinds ]. }
      apply (Aliased_loop K k params metadata metadata handler
        (bind_aliased_params handler) (hm_params_kinds metadata) aliases body
        (bind_aliased_params body)).
      * apply bind_aliased_params_aliased, safe_cons; [ exact Hhandler | | | ].
        -- intros i s Hs.
           destruct (Halias i s Hs)
             as (_ & _ & _ & _ & _ & param & Hparam & Hloop & _).
           apply (always_given_weaken k i
             (fun arg => simple_eqb arg s || simple_eqb arg (Var param)));
             [ | exact Hloop ].
           intros arg Harg. unfold accepts. simpl. rewrite Hparam.
           exact Harg.
        -- intros x (i & s & Hs & Hx).
           destruct (Halias i s Hs)
             as (_ & _ & _ & Hstable & _ & param & Hparam & _ & Hbinds).
           destruct Hx as [-> | Halt];
             [ exact (proj1 (proj2 (Hstable x eq_refl))) | ].
           simpl in Halt. rewrite Hparam in Halt. injection Halt as <-.
           exact Hbinds.
        -- intros i s Hs. exact (proj1 (Halias i s Hs)).
      * exact Hparams.
      * intros i s Hs.
        destruct (Halias i s Hs) as (_ & _ & Hnodup & Hstable & _).
        split; [ exact Hnodup | ]. intros x Hx. exact (proj1 (Hstable x Hx)).
      * reflexivity.
      * apply kinds_fit_refl.
      * apply bind_aliased_params_binds_statics.
      * apply bind_aliased_params_aliased, safe_cons; [ exact Hbody | | | ].
        -- intros i s Hs.
           destruct (Halias i s Hs) as (_ & _ & _ & _ & Hgiven & _).
           apply (always_given_weaken k i (fun arg => simple_eqb arg s));
             [ | exact Hgiven ].
           intros arg Harg. unfold accepts. rewrite Harg. reflexivity.
        -- intros x (i & s & Hs & [-> | Halt]);
             [ | simpl in Halt; destruct i; discriminate Halt ].
           destruct (Halias i (Var x) Hs) as (_ & _ & _ & Hstable & _).
           exact (proj2 (proj2 (Hstable x eq_refl))).
        -- intros i s Hs. exact (proj1 (proj2 (Halias i s Hs))).
    + (* nothing is said of the continuations *)
      simpl. rewrite Halone.
      assert (Hall :
        Forall
          (fun kh =>
             safe (plain_fates handlers ++ K) (handler_expr (snd kh))
             ∧ ∀ p : variable, p ∈ handler_params (snd kh) -> unfrozen K p)
          handlers).
      { apply Forall_forall. intros [k [params metadata handler]] Hin.
        destruct (Hhandlers k params metadata handler Hin)
          as [Hhandler Hparams].
        split; [ apply safe_plain, Hhandler | exact Hparams ]. }
      apply Aliased_let_cont_recursive.
      * clear Hsafe Hhandlers Halone. revert Hall.
        generalize (plain_fates handlers ++ K)
          (Let_cont_recursive handlers body).
        intros K' whole Hall.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          [ apply Forall2_nil | ].
        inversion Hall as [ | kh rest [Hhandler Hparams] Hrest]. subst kh rest.
        apply Forall2_cons; [ | exact (IH Hrest) ].
        simpl. rewrite bind_aliases_nil.
        split; [ reflexivity | ]. split; [ reflexivity | ].
        split; [ reflexivity | ].
        split; [ apply bind_aliased_params_aliased, Hhandler | exact Hparams ].
      * clear. unfold handlers_bind_statics.
        generalize (Let_cont_recursive handlers body). intros whole.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ reflexivity | ].
        rewrite bind_aliases_binds_statics, bind_aliased_params_binds_statics,
          IH.
        reflexivity.
      * apply bind_aliased_params_aliased, safe_plain, Hbody.
  - apply Aliased_apply.
    + intros i s Hs. destruct (Hsafe (ap_continuation ap)) as (_ & _ & Hret).
      specialize (Hret i s Hs). simpl in Hret.
      rewrite Pos.eqb_refl in Hret. discriminate Hret.
    + intros i s Hs.
      destruct (Hsafe (ap_exn_continuation ap)) as (_ & _ & Hret).
      specialize (Hret i s Hs). simpl in Hret.
      rewrite Pos.eqb_refl, orb_true_r in Hret. discriminate Hret.
  - apply Aliased_apply_cont. intros i s Hs.
    destruct (Hsafe (ac_continuation ac)) as (Hgiven & _).
    specialize (Hgiven i s Hs). simpl in Hgiven.
    exact (given_fits _ _ _ _ _ Hgiven eq_refl).
  - apply Aliased_switch. intros arm Hin i s Hs.
    destruct (Hsafe (ac_continuation (snd arm))) as (Hgiven & _).
    specialize (Hgiven i s Hs). simpl in Hgiven.
    rewrite forallb_forall in Hgiven.
    exact (given_fits _ _ _ _ _ (Hgiven arm Hin) eq_refl).
  - apply Aliased_invalid.
Qed.

Theorem bind_aliased_params_refines : bind_aliased_params_refines_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (stuttering_refines state_aliased same_values aliased_step).
  - intros last vs vs' Hsame. exact Hsame.
  - apply (State_aliased []);
      [ apply bind_aliased_params_aliased, safe_nil | reflexivity
      | intros k; apply Same_aliased, no_alias_nil ].
Qed.
